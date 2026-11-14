import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import assistant, llm
from app.main import app


class CustomerSessionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_api_ask_requires_customer_session_and_rejects_body_identity(self):
        response = self.client.post(
            "/api/ask", json={"question": "Де моє замовлення?"}
        )
        self.assertEqual(response.status_code, 401)

        self.client.post("/api/session", json={"customer_id": "C-1001"})
        spoofed = self.client.post(
            "/api/ask",
            json={"customer_id": "C-1002", "question": "Чуже замовлення"},
        )
        self.assertEqual(spoofed.status_code, 422)

    def test_signed_session_identity_is_used_for_the_assistant(self):
        self.client.post("/api/session", json={"customer_id": "C-1002"})
        with patch.object(
            assistant,
            "handle_customer_query",
            return_value={"answer": "Готово", "calls": []},
        ) as handle:
            response = self.client.post(
                "/api/ask",
                json={"question": "Мої замовлення?", "mode": "api"},
            )

        self.assertEqual(response.status_code, 200)
        handle.assert_called_once_with("C-1002", "Мої замовлення?", "api")

    def test_session_rejects_unknown_customer(self):
        response = self.client.post(
            "/api/session", json={"customer_id": "C-9999"}
        )
        self.assertEqual(response.status_code, 404)

    def test_shop_outage_returns_503_with_tool_trace(self):
        self.client.post("/api/session", json={"customer_id": "C-1001"})
        payload = {
            "answer": "Сервіс недоступний.",
            "calls": [{"name": "check_stock", "status": "error"}],
            "rounds": 1,
            "stopped": "shop_unavailable",
            "model": "test-model",
            "elapsed": {"model": 0.1, "tools": 0.01},
            "usage": {"total_tokens": 2},
        }
        with patch.object(
            assistant,
            "handle_customer_query",
            side_effect=assistant.ShopUnavailableError(payload),
        ):
            response = self.client.post(
                "/api/ask", json={"question": "Є товар?"}
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["calls"], payload["calls"])

    def test_missing_model_key_returns_configuration_status_not_500(self):
        self.client.post("/api/session", json={"customer_id": "C-1001"})
        with (
            patch.object(llm, "API_KEY", None),
            patch.object(llm, "client", None),
        ):
            response = self.client.post(
                "/api/ask",
                json={"question": "Де замовлення?", "mode": "api"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("LLM_API_KEY", response.json()["detail"])

    def test_shop_timeout_returns_504_with_trace(self):
        self.client.post("/api/session", json={"customer_id": "C-1001"})
        payload = {
            "answer": "Сервіс не відповів.",
            "calls": [{"name": "check_stock", "status": "error"}],
            "rounds": 1,
            "stopped": "shop_timeout",
            "model": "test-model",
            "elapsed": {"model": 0.1, "tools": 0},
            "usage": {"total_tokens": 2},
        }
        with patch.object(
            assistant,
            "handle_customer_query",
            side_effect=assistant.ShopTimeoutError(payload),
        ):
            response = self.client.post(
                "/api/ask", json={"question": "Є товар?"}
            )

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["calls"], payload["calls"])

    def test_model_provider_timeout_returns_504(self):
        response = self.client.post(
            "/api/session", json={"customer_id": "C-1001"}
        )
        self.assertEqual(response.status_code, 200)
        with patch.object(
            assistant,
            "handle_customer_query",
            side_effect=llm.LLMProviderTimeoutError("HTTP 504 from provider"),
        ):
            response = self.client.post(
                "/api/ask",
                json={"question": "Статус замовлення?", "mode": "api"},
            )

        self.assertEqual(response.status_code, 504)
        self.assertIn("HTTP 504", response.json()["detail"])

    def test_local_mock_api_request_needs_no_live_model(self):
        from app import llm

        self.client.post("/api/session", json={"customer_id": "C-1002"})
        with (
            patch.object(llm, "MOCK_LLM", False),
            patch.object(llm, "get_client") as create_client,
        ):
            response = self.client.post(
                "/api/ask",
                json={"question": "Де моє замовлення 1002?", "mode": "mock"},
            )
            create_client.assert_not_called()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "mock")
        self.assertIn("Замовлення поточного демо-клієнта", payload["answer"])
        self.assertEqual(payload["calls"][0]["name"], "list_orders")
        self.assertIn("10466", payload["answer"])

    def test_ollama_mode_is_accepted_by_api(self):
        self.client.post("/api/session", json={"customer_id": "C-1001"})
        with patch.object(
            assistant,
            "handle_customer_query",
            return_value={"answer": "4", "mode": "ollama"},
        ) as handle:
            response = self.client.post(
                "/api/ask",
                json={"question": "Скільки буде 2 + 2?", "mode": "ollama"},
            )

        self.assertEqual(response.status_code, 200)
        handle.assert_called_once_with("C-1001", "Скільки буде 2 + 2?", "ollama")



if __name__ == "__main__":
    unittest.main()
