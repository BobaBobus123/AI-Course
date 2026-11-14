import unittest

from app.llm import LLMRateLimitError, _provider_error_message


class ProviderErrorTests(unittest.TestCase):
    def test_quota_errors_are_actionable(self):
        message = _provider_error_message(429)
        self.assertIn("квоту", message)
        self.assertIn("ліміт", message)

    def test_authentication_errors_are_actionable(self):
        message = _provider_error_message(401)
        self.assertIn("API-ключ", message)
        self.assertIn("LLM_API_KEY", message)

    def test_missing_model_errors_are_actionable(self):
        message = _provider_error_message(404)
        self.assertIn("LLM_MODEL_NAME", message)
        self.assertIn("LLM_BASE_URL", message)

    def test_rate_limit_has_distinct_api_error_type(self):
        error = LLMRateLimitError(_provider_error_message(429))
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("ліміт", str(error))


if __name__ == "__main__":
    unittest.main()
