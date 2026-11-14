import unittest

from eval.run_evaluation import (
    _evaluate_fields,
    _load_cases,
    _normalized,
)


class EvaluationTests(unittest.TestCase):
    def test_numeric_and_iban_normalization(self):
        self.assertEqual(_normalized("total_to_pay", "12 570,00"), _normalized("total_to_pay", "12570.00"))
        self.assertEqual(_normalized("supplier.iban", "UA66 3510 0500"), _normalized("supplier.iban", "ua6635100500"))

    def test_field_results_distinguish_omission_error_and_hallucination(self):
        results = _evaluate_fields(
            {
                "iban": "UA123",
                "missing": "known",
                "expected_null": None,
            },
            {
                "iban": "UA456",
                "missing": None,
                "expected_null": "invented",
            },
        )

        self.assertEqual(
            {item["field"]: item["label"] for item in results},
            {
                "expected_null": "hallucination",
                "iban": "error",
                "missing": "omission",
            },
        )

    def test_sample_manifest_loads_expected_source(self):
        cases = _load_cases(include_own=True)
        own_cases = [case for case in cases if case[0].startswith("own/")]

        self.assertEqual(len(own_cases), 1)
        self.assertEqual(
            own_cases[0][2]["expected_source"], "clean/rahunok-04.png"
        )


if __name__ == "__main__":
    unittest.main()
