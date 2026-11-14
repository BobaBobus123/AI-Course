import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import APIStatusError, APITimeoutError

from app import llm


class LLMRequestTests(unittest.TestCase):
    def test_local_mock_creates_tool_call_without_api_client(self):
        with (
            patch.object(llm, "MOCK_LLM", False),
            patch.object(llm, "get_client") as create_client,
        ):
            result = llm.chat(
                llm.build_messages("Де моє замовлення 10458?"),
                mode="mock",
            )

        create_client.assert_not_called()
        self.assertTrue(result["mock"])
        self.assertEqual(result["model"], "local-mock")
        self.assertEqual(result["usage"]["total_tokens"], 0)
        tool_call = result["message"].tool_calls[0]
        self.assertEqual(tool_call.function.name, "get_order")
        self.assertEqual(
            json.loads(tool_call.function.arguments), {"order_id": "10458"}
        )

    def test_mock_does_not_treat_customer_number_as_an_order_id(self):
        selected = llm._mock_tool_for_query(
            "Покажи замовлення 1002", customer_id="C-1002"
        )

        self.assertEqual(selected, ("list_orders", {}))

    def test_mock_ignores_customer_id_suffix_when_extracting_order_number(self):
        selected = llm._mock_tool_for_query(
            "Покажи мої замовлення для клієнта C-1002",
            customer_id="C-1002",
        )

        self.assertEqual(selected, ("list_orders", {}))

    def test_mock_routes_russian_product_and_stock_questions(self):
        product = llm._mock_tool_for_query(
            "Сколько стоит робот-пылесос Сириус?"
        )
        stock = llm._mock_tool_for_query(
            "Есть ли в наличии робот-пылесос Сириус?"
        )

        self.assertEqual(product, ("search_catalog", {
            "query": "робот-пилосос сіріус",
            "limit": 5,
        }))
        self.assertEqual(stock, product)

    def test_mock_extracts_product_without_question_intent_and_location_words(self):
        selected = llm._mock_tool_for_query(
            "Чи є в наявності роутер Альтаїр AX3 і скільки "
            "коштуватиме доставка у Львів у відділення?"
        )

        self.assertEqual(
            selected,
            ("search_catalog", {"query": "роутер альтаїр ax3", "limit": 5}),
        )

    def test_chat_sends_allowlisted_tools_and_auto_tool_choice(self):
        message = SimpleNamespace(content="Привіт", tool_calls=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            model="test-model",
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )
        create = Mock(return_value=response)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        with (
            patch.object(llm, "MOCK_LLM", True),
            patch.object(llm, "client", fake_client),
        ):
            result = llm.chat(
                llm.build_messages("Чи є товар у наявності?"), mode="api"
            )

        self.assertEqual(result["message"], message)
        self.assertEqual(result["model"], "test-model")
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], "auto")
        self.assertTrue(kwargs["tools"])
        self.assertEqual(kwargs["max_tokens"], llm.MAX_OUTPUT_TOKENS)
        self.assertEqual(kwargs["messages"][0]["role"], "system")

    def test_general_question_does_not_send_shop_tools(self):
        message = SimpleNamespace(content="Антуан де Сент-Екзюпері", tool_calls=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            model="test-model",
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )
        create = Mock(return_value=response)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        with patch.object(llm, "get_client", return_value=fake_client):
            llm.chat(
                llm.build_messages("Хто написав повість «Маленький принц»?"),
                mode="api",
            )

        self.assertNotIn("tools", create.call_args.kwargs)
        self.assertNotIn("tool_choice", create.call_args.kwargs)

    def test_ollama_mode_uses_local_client_and_configured_model(self):
        message = SimpleNamespace(content="Привіт", tool_calls=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            model=llm.OLLAMA_MODEL,
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
        )
        create = Mock(return_value=response)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        with (
            patch.object(llm, "API_KEY", None),
            patch.object(llm, "get_client") as api_client,
            patch.object(llm, "get_ollama_client", return_value=fake_client),
        ):
            result = llm.chat(
                llm.build_messages("Чи є товар у наявності?"), mode="ollama"
            )

        api_client.assert_not_called()
        self.assertEqual(result["model"], llm.OLLAMA_MODEL)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], llm.OLLAMA_MODEL)
        self.assertEqual(kwargs["tool_choice"], "auto")

    def test_local_client_timeout_has_distinct_message(self):
        request = httpx.Request("POST", "https://example.test")
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(side_effect=APITimeoutError(request=request))
                )
            )
        )

        with (
            patch.object(llm, "client", fake_client),
            self.assertRaises(llm.LLMTimeoutError) as raised,
        ):
            llm.chat([], mode="api")

        self.assertIn("локальний таймаут", str(raised.exception))

    def test_provider_http_504_has_distinct_message(self):
        request = httpx.Request("POST", "https://example.test")
        response = httpx.Response(504, request=request)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        side_effect=APIStatusError(
                            "upstream timeout", response=response, body=None
                        )
                    )
                )
            )
        )

        with (
            patch.object(llm, "client", fake_client),
            self.assertRaises(llm.LLMProviderTimeoutError) as raised,
        ):
            llm.chat([], mode="api")

        self.assertIn("HTTP 504", str(raised.exception))

    def test_provider_http_503_is_reported_as_upstream_status(self):
        request = httpx.Request("POST", "https://example.test")
        response = httpx.Response(503, request=request)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        side_effect=APIStatusError(
                            "upstream unavailable", response=response, body=None
                        )
                    )
                )
            )
        )

        with (
            patch.object(llm, "client", fake_client),
            self.assertRaises(llm.LLMUnavailableError) as raised,
        ):
            llm.chat([], mode="api")

        self.assertIn("HTTP 503", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
