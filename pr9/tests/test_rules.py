import unittest
from datetime import date
from unittest.mock import patch

from app import rules


class RequestRulesTests(unittest.TestCase):
    def test_detects_two_actions_but_not_one(self):
        self.assertTrue(rules.detect_multiple("Скасуйте замовлення і поверніть товар"))
        self.assertFalse(rules.detect_multiple("Скасуйте замовлення"))

    def test_infers_return_reason_from_customer_text(self):
        self.assertEqual(rules.reason_from_text("Не заряджається", None), "defect")
        self.assertEqual(rules.reason_from_text("Пошкодили під час доставки", None), "damaged")
        self.assertEqual(rules.reason_from_text("Завеликий", None), "not_suitable")
        self.assertEqual(rules.reason_from_text("Поверніть", "wrong_item"), "wrong_item")

    def test_matches_item_only_when_unambiguous(self):
        robot = {"sku": "SIR-R10", "name": "Робот пилосос"}
        vacuum = {"sku": "SIR-V20", "name": "Пилосос"}
        order = {"items": [robot, vacuum]}
        self.assertIs(rules.match_item(order, "SIR-R10"), robot)
        self.assertIsNone(rules.match_item(order, "пилосос"))
        self.assertIsNone(rules.match_item(order, None))
        self.assertIs(rules.match_item({"items": [robot]}, None), robot)

    @patch("app.rules.service.today", return_value=date(2026, 10, 6))
    def test_return_policy_preserves_delivery_and_time_limits(self, _today):
        order = {
            "status": "delivered",
            "delivery": {"delivered_at": "2026-10-01"},
        }
        product = {"returnable": True, "warranty_months": 1}
        self.assertIsNone(rules.return_policy_problem(order, product, "not_suitable"))
        self.assertEqual(
            rules.return_policy_problem(
                {"status": "shipped", "delivery": order["delivery"]},
                product,
                "not_suitable",
            ),
            "Повернення оформлюється лише для отриманого замовлення.",
        )
        self.assertIn(
            "строк повернення",
            rules.return_policy_problem(
                {**order, "delivery": {"delivered_at": "2026-09-01"}},
                product,
                "not_suitable",
            ).lower(),
        )
        self.assertEqual(
            rules.return_policy_problem(
                order,
                {**product, "returnable": False},
                "not_suitable",
            ),
            "Цей товар належної якості не підлягає поверненню.",
        )
        self.assertEqual(
            rules.return_policy_problem(
                {**order, "delivery": {"delivered_at": "2026-08-01"}},
                product,
                "defect",
            ),
            "Гарантійний строк минув.",
        )


if __name__ == "__main__":
    unittest.main()
