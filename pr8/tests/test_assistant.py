import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import assistant, tools


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=True):
        data = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return data


def model_result(message, prompt_tokens=4, completion_tokens=2):
    return {
        "message": message,
        "elapsed": 0.01,
        "model": "test-model",
        "usage": SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    }


class AssistantCycleTests(unittest.TestCase):
    def test_mock_runs_real_tool_validation_and_returns_order_status(self):
        with patch.object(assistant.llm, "MOCK_LLM", True):
            result = assistant.handle_customer_query(
                "C-1001", "Де моє замовлення 10458?"
            )

        self.assertEqual(result["mode"], "mock")
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["calls"][0]["name"], "get_order")
        self.assertEqual(result["calls"][0]["status"], "ok")
        self.assertIn("Передано в доставку", result["answer"])
        self.assertIn("20450012345678", result["answer"])

    def test_mock_answers_order_lists_for_each_demo_customer(self):
        expected_orders = {
            "C-1001": {"10421", "10458", "10463"},
            "C-1002": {"10377", "10466"},
            "C-1003": {"10412", "10470"},
        }
        for customer_id, order_ids in expected_orders.items():
            with self.subTest(customer_id=customer_id), patch.object(
                assistant.llm, "MOCK_LLM", True
            ):
                result = assistant.handle_customer_query(
                    customer_id, "Покажи мои заказы"
                )
            self.assertEqual(result["mode"], "mock")
            self.assertEqual(result["calls"][0]["name"], "list_orders")
            for order_id in order_ids:
                self.assertIn(order_id, result["answer"])
            other_ids = set().union(*expected_orders.values()) - order_ids
            for order_id in other_ids:
                self.assertNotIn(order_id, result["answer"])

    def test_mock_understands_stock_question_by_product_name_in_russian(self):
        with patch.object(assistant.llm, "MOCK_LLM", True):
            result = assistant.handle_customer_query(
                "C-1001", "Есть ли в наличии робот-пылесос Сириус?"
            )

        self.assertEqual(
            [call["name"] for call in result["calls"]],
            ["search_catalog", "check_stock"],
        )
        self.assertIn("товару SIR-R10", result["answer"])

    def test_mock_answers_combined_stock_and_delivery_question(self):
        with patch.object(assistant.llm, "MOCK_LLM", True):
            result = assistant.handle_customer_query(
                "C-1002",
                "Чи є в наявності роутер Альтаїр AX3 і скільки "
                "коштуватиме доставка у Львів у відділення?",
            )

        self.assertEqual(
            [call["name"] for call in result["calls"]],
            ["search_catalog", "check_stock", "calculate_delivery"],
        )
        self.assertEqual(result["rounds"], 3)
        self.assertEqual(result["calls"][2]["status"], "ok")
        self.assertEqual(
            result["calls"][2]["arguments"]["city"], "Львів"
        )
        self.assertEqual(
            result["calls"][2]["arguments"]["method"], "branch"
        )
        self.assertIn("7 шт.", result["answer"])
        self.assertIn("0 грн (безкоштовно)", result["answer"])
        self.assertIn("1–3 робочі дні", result["answer"])
        self.assertIn("від 2000.00 грн", result["answer"])
        self.assertIn("Альтаїр AX3", result["answer"])
        self.assertIn("3299.00 грн", result["answer"])
        self.assertNotIn("None", result["answer"])

    def test_mock_answers_store_policy_and_clarifies_incomplete_return(self):
        with patch.object(assistant.llm, "MOCK_LLM", True):
            policy = assistant.handle_customer_query(
                "C-1002", "Сколько дней можно вернуть товар?"
            )
            incomplete_return = assistant.handle_customer_query(
                "C-1002", "Хочу оформить возврат"
            )

        self.assertIn("14 днів", policy["answer"])
        self.assertEqual(policy["calls"], [])
        self.assertIn("номер свого замовлення", incomplete_return["answer"])
        self.assertEqual(incomplete_return["calls"], [])

    def test_tool_call_and_tool_result_are_appended_to_model_history(self):
        call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="get_order", arguments='{"order_id":"10421"}'
            ),
        )
        responses = [
            model_result(FakeMessage(tool_calls=[call])),
            model_result(FakeMessage(content="Замовлення доставлено.")),
        ]
        histories = []

        def fake_chat(messages, mode=None, customer_id=None):
            histories.append(list(messages))
            return responses.pop(0)

        with patch.object(assistant.llm, "chat", side_effect=fake_chat):
            result = assistant.handle_customer_query(
                "C-1001", "Де замовлення 10421?", mode="api"
            )

        self.assertEqual(result["answer"], "Замовлення доставлено.")
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["usage"]["prompt_tokens"], 8)
        self.assertEqual(result["calls"][0]["status"], "ok")
        self.assertEqual(result["calls"][0]["action"],
                         "verified order ownership and filtered private fields")
        self.assertEqual(histories[1][-1]["role"], "tool")
        self.assertEqual(histories[1][-1]["tool_call_id"], "call-1")

    def test_unrequested_catalog_filters_do_not_hide_matching_products(self):
        call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="search_catalog",
                arguments=json.dumps({
                    "query": "Альтаїр AX3",
                    "category": "electronic",
                    "max_price": "1000",
                    "limit": 1,
                }),
            ),
        )
        responses = [
            model_result(FakeMessage(tool_calls=[call])),
            model_result(FakeMessage(content="Знайдено роутер Альтаїр AX3.")),
        ]
        with patch.object(assistant.llm, "chat", side_effect=responses):
            result = assistant.handle_customer_query(
                "C-1002", "Знайди роутер Альтаїр AX3", mode="api"
            )

        self.assertEqual(result["calls"][0]["status"], "ok")
        self.assertEqual(result["calls"][0]["result"][0]["sku"], "ALT-AX3")
        self.assertNotIn("category", result["calls"][0]["arguments"])
        self.assertNotIn("max_price", result["calls"][0]["arguments"])

    def test_tool_round_limit_is_enforced(self):
        call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="list_orders", arguments="{}"),
        )
        with (
            patch.object(assistant, "TOOL_MAX_ROUNDS", 1),
            patch.object(
                assistant.llm, "chat",
                return_value=model_result(FakeMessage(tool_calls=[call])),
            ),
        ):
            result = assistant.handle_customer_query(
                "C-1001", "Мої замовлення", mode="api"
            )

        self.assertEqual(result["stopped"], "limit")
        self.assertEqual(result["rounds"], 1)

    def test_shop_outage_preserves_trace_and_stops_before_model_retry(self):
        unavailable = tools.ToolResult(
            name="check_stock",
            arguments={"sku": "ORN-X2"},
            status="error",
            reason="unavailable",
            result={"error": "temporarily unavailable", "code": "unavailable"},
            content=json.dumps(
                {"error": "temporarily unavailable", "code": "unavailable"}
            ),
            action="shop service returned an error",
        )
        with (
            patch.object(
                assistant.llm,
                "chat",
                return_value=model_result(
                    FakeMessage(
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="check_stock",
                                    arguments='{"sku":"ORN-X2"}',
                                ),
                            )
                        ]
                    )
                ),
            ),
            patch.object(assistant.tools, "call", return_value=unavailable),
        ):
            with self.assertRaises(assistant.ShopUnavailableError) as raised:
                assistant.handle_customer_query(
                    "C-1001", "Є в наявності?", mode="api"
                )

        self.assertEqual(raised.exception.payload["stopped"], "shop_unavailable")
        self.assertEqual(len(raised.exception.payload["calls"]), 1)

    def test_shop_timeout_is_classified_separately(self):
        timeout = tools.ToolResult(
            name="check_stock",
            arguments={"sku": "ORN-X2"},
            status="error",
            reason="timed out",
            result={"error": "timed out", "code": "timeout"},
            content='{"error":"timed out","code":"timeout"}',
            action="not executed",
        )
        with (
            patch.object(
                assistant.llm,
                "chat",
                return_value=model_result(
                    FakeMessage(
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="check_stock",
                                    arguments='{"sku":"ORN-X2"}',
                                ),
                            )
                        ]
                    )
                ),
            ),
            patch.object(assistant.tools, "call", return_value=timeout),
        ):
            with self.assertRaises(assistant.ShopTimeoutError):
                assistant.handle_customer_query(
                    "C-1001", "Є товар?", mode="api"
                )


if __name__ == "__main__":
    unittest.main()
