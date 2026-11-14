import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import llm


class LLMApiTests(unittest.TestCase):
    def test_api_receives_text_and_image_and_schema_is_validated(self):
        response_data = {
            "doc_type": "рахунок на оплату",
            "number": "API-01",
            "date": "2026-01-15",
            "supplier": {"name": "Постачальник"},
            "buyer": {"name": "Покупець"},
            "items": [],
        }
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(response_data), refusal=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=40, total_tokens=140
            ),
        )
        client = Mock()
        client.chat.completions.create.return_value = completion

        with (
            patch.object(llm, "MOCK_LLM", False),
            patch.object(llm, "_create_client", return_value=client),
        ):
            invoice, metadata = llm.extract_invoice_data(
                "data:image/jpeg;base64,ZmFrZQ==", max_retries=0
            )

        request = client.chat.completions.create.call_args.kwargs
        image_part = request["messages"][1]["content"][1]
        self.assertEqual(image_part["type"], "image_url")
        self.assertIn("data:image/jpeg;base64,", image_part["image_url"]["url"])
        system_prompt = request["messages"][0]["content"]
        self.assertIn('"doc_type"', system_prompt)
        self.assertIn('"total_to_pay"', system_prompt)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(invoice.number, "API-01")
        self.assertEqual(metadata["total_tokens"], 140)
        self.assertFalse(metadata["mock"])

    def test_parser_rejects_invalid_json_schema(self):
        with self.assertRaises(llm.LLMResponseError):
            llm._parse_invoice_content('{"unexpected":"value"}')

    def test_parser_accepts_json_in_text_content_blocks(self):
        content = [
            {"type": "text", "text": '{"doc_type":"рахунок на оплату",'},
            {"type": "text", "text": '"number":"1","supplier":{},"buyer":{}}'},
        ]

        invoice = llm._parse_invoice_content(content)

        self.assertEqual(invoice.number, "1")
        self.assertEqual(invoice.items, [])


if __name__ == "__main__":
    unittest.main()
