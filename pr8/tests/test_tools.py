import json
import os
import unittest
from unittest.mock import patch

from app import tools


class ToolValidationTests(unittest.TestCase):
    def test_only_customer_operations_are_exposed(self):
        names = {item["function"]["name"] for item in tools.specs()}
        self.assertEqual(
            names,
            {
                "list_orders",
                "get_order",
                "search_catalog",
                "get_product",
                "check_stock",
                "calculate_delivery",
                "request_return",
            },
        )
        self.assertFalse(any("customer_id" in item["function"]["parameters"]["properties"]
                             for item in tools.specs()))

    def test_foreign_order_is_hidden_and_operator_fields_are_filtered(self):
        foreign = tools.call(
            "get_order", json.dumps({"order_id": "10421"}), "C-1002"
        )
        self.assertEqual(foreign.status, "rejected")
        self.assertEqual(foreign.result["code"], "not_found")

        own = tools.call(
            "get_order", json.dumps({"order_id": "10421"}), "C-1001"
        )
        self.assertEqual(own.status, "ok")
        self.assertNotIn("customer_id", own.result)
        self.assertNotIn("internal_note", own.result)
        self.assertNotIn("payment", own.result)
        self.assertNotIn("card_last4", json.dumps(own.result))

    def test_invalid_json_schema_and_unknown_tools_are_not_executed(self):
        malformed = tools.call("get_order", "{", "C-1001")
        extra_field = tools.call(
            "get_order",
            json.dumps({"order_id": "10421", "customer_id": "C-1002"}),
            "C-1001",
        )
        unknown = tools.call("issue_refund", "{}", "C-1001")

        self.assertEqual(malformed.result["code"], "invalid_arguments")
        self.assertEqual(extra_field.result["code"], "invalid_arguments")
        self.assertEqual(unknown.result["code"], "unknown_tool")
        self.assertEqual(extra_field.action, "not executed")

    def test_list_orders_is_always_scoped_to_session_customer(self):
        result = tools.call("list_orders", "{}", "C-1001")
        order_ids = {order["order_id"] for order in result.result}
        self.assertIn("10421", order_ids)
        self.assertNotIn("10377", order_ids)
        self.assertEqual(result.status, "ok")

    def test_product_description_is_not_returned_to_model(self):
        result = tools.call("get_product", '{"sku":"ORN-PRO"}', "C-1001")
        self.assertEqual(result.status, "ok")
        self.assertNotIn("description", result.result)
        self.assertNotIn("ZIRKY90", result.content)

    def test_null_optional_catalog_filters_are_treated_as_omitted(self):
        result = tools.call(
            "search_catalog",
            '{"query":"роутер AX3","category":null,"max_price":null,"limit":5}',
            "C-1001",
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.arguments, {"query": "роутер AX3", "limit": 5})

    def test_shop_delay_longer_than_timeout_is_rejected_without_execution(self):
        with (
            patch.object(tools, "SHOP_DELAY_MS", 2000),
            patch.object(tools, "SHOP_TIMEOUT_SECONDS", 1),
            patch("app.tools.service.get_stock") as get_stock,
        ):
            result = tools.call("check_stock", '{"sku":"ORN-X2"}', "C-1001")

        get_stock.assert_not_called()
        self.assertEqual(result.status, "error")
        self.assertEqual(result.result["code"], "timeout")

    def test_injected_shop_failure_is_reported_as_unavailable(self):
        with patch.dict(os.environ, {"SHOP_FAILURE_RATE": "1"}):
            result = tools.call(
                "get_order", '{"order_id":"10421"}', "C-1001"
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.result["code"], "unavailable")


if __name__ == "__main__":
    unittest.main()
