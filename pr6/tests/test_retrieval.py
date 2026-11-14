import unittest

from app.retrieval import _fit_context_block


class FitContextBlockTests(unittest.TestCase):
    def test_truncates_long_passage_to_context_budget(self):
        chunk = {
            "metadata": {"title": "Доставка", "updated": "2026-01-01"},
            "section": "Відправлення",
            "raw_text": "Інформація про відправлення замовлення. " * 80,
        }

        fitted = _fit_context_block(chunk, source_id=1, max_tokens=60)

        self.assertIsNotNone(fitted)
        block, passage, token_count = fitted
        self.assertLessEqual(token_count, 60)
        self.assertTrue(passage)
        self.assertIn("[1] Документ: Доставка", block)


if __name__ == "__main__":
    unittest.main()
