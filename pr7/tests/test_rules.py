import json
import unittest
from pathlib import Path

from app.extraction import decide
from app.rules import (
    is_valid_edrpou,
    is_valid_iban,
    is_valid_rnokpp,
    validate_invoice,
)
from app.schema import InvoiceSchema


class ChecksumTests(unittest.TestCase):
    def test_reference_checksums_are_valid(self):
        self.assertTrue(is_valid_edrpou("38124779"))
        self.assertTrue(is_valid_edrpou("44172914"))
        self.assertTrue(is_valid_rnokpp("3128704592"))
        self.assertTrue(is_valid_iban("UA663510050000026004017723561"))

    def test_tampered_checksums_are_invalid(self):
        self.assertFalse(is_valid_edrpou("38124770"))
        self.assertFalse(is_valid_rnokpp("3128704593"))
        self.assertFalse(is_valid_iban("UA663510050000026004017723560"))


class InvoiceValidationTests(unittest.TestCase):
    def valid_invoice(self):
        return InvoiceSchema.model_validate(
            {
                "doc_type": "рахунок на оплату",
                "number": "INV-01",
                "date": "2026-09-03",
                "supplier": {
                    "name": "ТОВ «Оріон Аудіо Дистрибуція»",
                    "code": "38124779",
                    "iban": "UA663510050000026004017723561",
                },
                "buyer": {
                    "name": "ТОВ «Сузірʼя Рітейл»",
                    "code": "44172914",
                },
                "items": [
                    {
                        "name": "Навушники",
                        "unit": "шт",
                        "quantity": "2",
                        "price": "100.00",
                        "sum": "200.00",
                    }
                ],
                "total_no_vat": "200.00",
                "vat": "40.00",
                "total_to_pay": "240.00",
            }
        )

    def test_valid_invoice_has_no_issues_and_can_be_auto_approved(self):
        invoice = self.valid_invoice()
        issues = validate_invoice(invoice)

        self.assertEqual(issues, [])
        self.assertEqual(decide(invoice.model_dump(), issues), ("auto", []))

    def test_missing_critical_fields_require_review(self):
        invoice = self.valid_invoice()
        invoice.number = None
        invoice.buyer.code = None
        invoice.items[0].sum = None
        issues = validate_invoice(invoice)

        self.assertTrue(issues)
        self.assertTrue(
            all(
                set(issue) == {"field", "rule", "message", "severity"}
                and issue["severity"] in {"error", "warning"}
                for issue in issues
            )
        )
        decision, reasons = decide(invoice.model_dump(), issues)
        self.assertEqual(decision, "review")
        self.assertTrue(reasons)

    def test_decimal_line_and_vat_mismatches_require_review(self):
        invoice = self.valid_invoice()
        invoice.items[0].price = "100.005"
        invoice.vat = "39.99"
        issues = validate_invoice(invoice)
        rules = {issue["rule"] for issue in issues}

        self.assertIn("line_math", rules)
        self.assertIn("vat_rate", rules)

    def test_non_invoice_is_rejected(self):
        invoice = self.valid_invoice().model_dump()
        invoice["doc_type"] = "видаткова накладна"
        self.assertEqual(decide(invoice, [])[0], "reject")

    def test_all_expected_sample_decisions_match_the_rules(self):
        project_dir = Path(__file__).resolve().parents[1]
        expected = json.loads(
            (project_dir / "samples" / "expected.json").read_text(encoding="utf-8")
        )

        for filename, record in expected.items():
            fields = record["fields"]
            invoice = InvoiceSchema.model_validate(
                {
                    "doc_type": fields.get("document_type"),
                    "number": fields.get("number"),
                    "date": fields.get("date"),
                    "due_date": fields.get("valid_until"),
                    "supplier": fields.get("supplier") or {},
                    "buyer": fields.get("buyer") or {},
                    "items": [
                        {
                            "name": item.get("name"),
                            "unit": item.get("unit"),
                            "quantity": item.get("quantity"),
                            "price": item.get("price"),
                            "sum": item.get("amount"),
                        }
                        for item in fields.get("items", [])
                    ],
                    "total_no_vat": fields.get("total_without_vat"),
                    "vat": fields.get("vat"),
                    "total_to_pay": fields.get("total"),
                }
            )
            decision = decide(
                invoice.model_dump(), validate_invoice(invoice)
            )[0]
            with self.subTest(sample=filename):
                self.assertEqual(decision, record["expected_decision"])


if __name__ == "__main__":
    unittest.main()
