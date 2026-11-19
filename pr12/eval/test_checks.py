import unittest
from types import SimpleNamespace
from unittest.mock import patch

import checks
import judge
import openai


class DeterministicCheckTests(unittest.TestCase):
    def run_one(self, spec, case, output):
        case = {**case, "checks": [spec]}
        return checks.run_checks(case, output)[0]

    def test_language_rejects_latin_mock_corruption(self):
        result = self.run_one(
            {"check": "language", "lang": "uk"},
            {"input": {}},
            {"text": "Your order is being delivered tomorrow."},
        )
        self.assertEqual(result.name, "language")
        self.assertFalse(result.passed)

    def test_language_accepts_ukrainian_with_latin_product_name(self):
        result = self.run_one(
            {"check": "language", "lang": "uk"},
            {"input": {}},
            {"text": "Ваш роутер ALT-AX3 вже їде до відділення."},
        )
        self.assertTrue(result.passed, result.detail)

    def test_language_rejects_russian_cyrillic(self):
        result = self.run_one(
            {"check": "language", "lang": "uk"},
            {"input": {}},
            {"text": "Ваш заказ уже отправлен и скоро прибудет."},
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.data["ukrainian_letters"], 0)

    def test_numbers_rejects_unfounded_mock_number(self):
        result = self.run_one(
            {"check": "numbers_from_facts"},
            {"input": {"text": "Де замовлення 10458?", "facts": ["Передано 04.10.2026."]}},
            {"text": "Замовлення 10458 прибуде за 3 дні."},
        )
        self.assertEqual(result.name, "numbers_from_facts")
        self.assertFalse(result.passed)
        self.assertIn("3", result.data["unexpected"])

    def test_numbers_accepts_grouped_number_and_fact_date_parts(self):
        result = self.run_one(
            {"check": "numbers_from_facts"},
            {"input": {"text": "Замовлення № 10 458", "facts": ["Передано 04.10.2026."]}},
            {"text": "Замовлення 10458 передано 4 жовтня 2026 року."},
        )
        self.assertTrue(result.passed, result.detail)

    def test_numbers_normalizes_dates_and_ukrainian_numerals(self):
        result = self.run_one(
            {"check": "numbers_from_facts"},
            {"input": {"facts": ["Повернення до 13.10; очікування до 7 днів."]}},
            {"text": "Повернення до 13 жовтня; очікування протягом семи днів."},
        )
        self.assertTrue(result.passed, result.detail)

    def test_numbers_allow_supports_extra_values(self):
        result = self.run_one(
            {"check": "numbers_from_facts", "allow": ["24"]},
            {"input": {"facts": ["Замовлення доставлено."]}},
            {"text": "Відповідь надійде за 24 години."},
        )
        self.assertTrue(result.passed, result.detail)

    def test_numbers_rejects_unallowed_number_word(self):
        result = self.run_one(
            {"check": "numbers_from_facts"},
            {"input": {"facts": ["Заявку оформлено."]}},
            {"text": "Відповідь надійде за сім днів."},
        )
        self.assertFalse(result.passed)
        self.assertIn("7", result.data["unexpected"])

    def test_system_prompt_check_rejects_contiguous_leak(self):
        prompt = "Ви оператор служби підтримки. Спирайтеся лише на факти оператора і не вигадуйте даних."
        result = self.run_one(
            {"check": "no_system_prompt", "min_run": 8},
            {"input": {}},
            {"text": "Спирайтеся лише на факти оператора і не вигадуйте даних.", "system_prompt": prompt},
        )
        self.assertEqual(result.name, "no_system_prompt")
        self.assertFalse(result.passed)

    def test_system_prompt_markers_respect_min_run(self):
        result = self.run_one(
            {"check": "no_system_prompt", "min_run": 4, "markers": ["службова інструкція для моделі"]},
            {"input": {}},
            {"text": "службова інструкція для моделі"},
        )
        self.assertFalse(result.passed)

    def test_args_match_checks_parsed_arguments_and_normalized_numbers(self):
        result = self.run_one(
            {"check": "args_match", "tool": "get_order", "arguments": {"order_id": 10458}},
            {"input": {}},
            {"calls": [{"name": "get_order", "status": "ok", "arguments": "{\"order_id\":\"10458\"}"}]},
        )
        self.assertEqual(result.name, "args_match")
        self.assertTrue(result.passed, result.detail)

    def test_args_match_rejects_invalid_json(self):
        result = self.run_one(
            {"check": "args_match", "tool": "get_order", "arguments": {"order_id": "10458"}},
            {"input": {}},
            {"calls": [{"name": "get_order", "status": "ok", "arguments": "{not-json"}]},
        )
        self.assertEqual(result.name, "args_match")
        self.assertFalse(result.passed)

    def test_args_match_prefers_raw_arguments_and_accepts_grouped_numbers(self):
        result = self.run_one(
            {"check": "args_match", "tool": "get_order", "arguments": {"order_id": "10458"}},
            {"input": {}},
            {"calls": [{
                "name": "get_order",
                "status": "ok",
                "arguments": {"order_id": "99999"},
                "raw_arguments": '{"order_id":"10 458"}',
            }]},
        )
        self.assertTrue(result.passed, result.detail)

    def test_args_match_optional_only_passes_when_tool_was_not_called(self):
        spec = {"check": "args_match", "tool": "get_order", "arguments": {"order_id": "10458"},
                "optional": True}
        self.assertTrue(self.run_one(spec, {"input": {}}, {"calls": []}).passed)
        rejected = self.run_one(
            spec, {"input": {}},
            {"calls": [{"name": "get_order", "status": "rejected", "arguments": "{}"}]},
        )
        self.assertFalse(rejected.passed)

    def test_check_errors_and_unknown_checks_are_case_local_failures(self):
        with patch.dict(checks.REGISTRY, {"broken": lambda *_: 1 / 0}):
            results = checks.run_checks(
                {"checks": [{"check": "unknown"}, {"check": "broken"}]},
                {"text": "valid"},
            )
        self.assertEqual([result.passed for result in results], [False, False])
        self.assertIn("невідома", results[0].detail)
        self.assertIn("ZeroDivisionError", results[1].detail)

    def test_invalid_judge_verdict_is_a_failed_check(self):
        case = {"input": {}, "checks": [{"check": "judge", "criteria": ["tone"]}]}
        output = {"text": "Добрий день.", "status": "ok"}
        with patch.object(judge, "evaluate", side_effect=ValueError("невалідний JSON")):
            result = checks.run_checks(case, output)[0]
        self.assertEqual(result.name, "judge")
        self.assertFalse(result.passed)
        self.assertIn("невалідний JSON", result.detail)

    def test_verdict_parser_rejects_missing_reason_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            judge.parse_verdict('{"criteria":[{"id":"tone","passed":true}],"overall":true}', ["tone"])
        duplicate = (
            '{"criteria":[{"id":"tone","passed":true,"reason":"ok"},'
            '{"id":"tone","passed":true,"reason":"ok"}],"overall":true}'
        )
        with self.assertRaises(ValueError):
            judge.parse_verdict(duplicate, ["tone"])

    def test_judge_uses_schema_and_retries_invalid_output_once(self):
        valid = '{"criteria":[{"id":"tone","passed":true,"reason":"Ввічливо."}],"overall":true}'
        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                raw = "not json" if len(calls) == 1 else valid
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=raw))],
                    usage=None,
                    model="judge-model",
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        with (
            patch.object(judge, "JUDGE_BASE_URL", "https://judge.invalid/v1"),
            patch.object(judge, "JUDGE_API_KEY", "not-a-real-key"),
            patch.object(judge, "JUDGE_MODEL", "judge-model"),
            patch.object(openai, "OpenAI", return_value=client),
        ):
            verdict = judge.evaluate(
                {"input": {"text": "Питання", "facts": []}},
                {"text": "Добрий день.", "model": "generator-model"},
                ["tone"],
            )

        self.assertTrue(verdict.overall)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["temperature"], 0)
        self.assertEqual(
            calls[0]["response_format"],
            {"type": "json_schema", "json_schema": {"name": "verdict", "schema": judge.verdict_schema()}},
        )
        self.assertEqual(calls[1]["messages"][-1]["role"], "user")

    def test_judge_falls_back_to_json_object_when_schema_is_rejected(self):
        valid = '{"criteria":[{"id":"tone","passed":true,"reason":"Ввічливо."}],"overall":true}'
        calls = []

        class UnsupportedSchema(Exception):
            pass

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if kwargs["response_format"]["type"] == "json_schema":
                    raise UnsupportedSchema()
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=valid))],
                    usage=None,
                    model="judge-model",
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        with (
            patch.object(judge, "JUDGE_BASE_URL", "https://judge.invalid/v1"),
            patch.object(judge, "JUDGE_API_KEY", "not-a-real-key"),
            patch.object(judge, "JUDGE_MODEL", "judge-model"),
            patch.object(openai, "OpenAI", return_value=client),
            patch.object(openai, "BadRequestError", UnsupportedSchema),
        ):
            verdict = judge.evaluate(
                {"input": {}}, {"text": "Добрий день.", "model": "generator-model"}, ["tone"]
            )

        self.assertTrue(verdict.overall)
        self.assertEqual(
            [call["response_format"]["type"] for call in calls],
            ["json_schema", "json_object"],
        )
        self.assertIn("JSON Schema", calls[1]["messages"][0]["content"])

    def test_judge_fails_after_two_invalid_responses(self):
        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))],
                    usage=None,
                    model="judge-model",
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        with (
            patch.object(judge, "JUDGE_BASE_URL", "https://judge.invalid/v1"),
            patch.object(judge, "JUDGE_API_KEY", "not-a-real-key"),
            patch.object(judge, "JUDGE_MODEL", "judge-model"),
            patch.object(openai, "OpenAI", return_value=client),
        ):
            result = checks.run_checks(
                {"input": {}, "checks": [{"check": "judge", "criteria": ["tone"]}]},
                {"text": "Добрий день.", "model": "generator-model"},
            )[0]

        self.assertFalse(result.passed)
        self.assertIn("збій судді після повтору", result.detail)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
