import unittest
from types import SimpleNamespace

from app.llm import _parse_rag_response


class ParseRagResponseTests(unittest.TestCase):
    def test_parses_plain_json(self):
        response = _parse_rag_response(
            SimpleNamespace(
                content=(
                    '{"response_text":"Перевірте статус замовлення [1].",'
                    '"found_in_base":true,"cited_source_ids":[1]}'
                )
            )
        )

        self.assertEqual(response.cited_source_ids, [1])
        self.assertTrue(response.found_in_base)

    def test_parses_json_surrounded_by_explanation_and_markdown(self):
        response = _parse_rag_response(
            SimpleNamespace(
                content=(
                    "Ось відповідь у потрібному форматі:\n```json\n"
                    '{"response_text":"Інформації немає.",'
                    '"found_in_base":false,"cited_source_ids":[]}\n```'
                )
            )
        )

        self.assertFalse(response.found_in_base)
        self.assertEqual(response.cited_source_ids, [])

    def test_extracts_typed_text_content_blocks(self):
        response = _parse_rag_response(
            SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text='{"response_text":"Немає даних.",'),
                    SimpleNamespace(
                        type="text",
                        text='"found_in_base":false,"cited_source_ids":[]}',
                    ),
                ]
            )
        )

        self.assertFalse(response.found_in_base)

    def test_rejects_json_missing_required_fields(self):
        with self.assertRaises(ValueError):
            _parse_rag_response(SimpleNamespace(content='{"answer":"Не знаю"}'))


if __name__ == "__main__":
    unittest.main()
