import unittest

from app.documents import chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_chunks_keep_their_markdown_section(self):
        document = {
            "filename": "returns.md",
            "metadata": {"title": "Returns"},
            "content": (
                "# Returns\n\n"
                "## Good-quality returns\n\n"
                "Customers have 14 days to return an unused item.\n\n"
                "## Return shipping\n\n"
                "The customer pays shipping for a good-quality return."
            ),
        }

        chunks = chunk_text(document)
        deadline_chunk = next(
            chunk for chunk in chunks if "14 days" in chunk["raw_text"]
        )
        shipping_chunk = next(
            chunk for chunk in chunks if "pays shipping" in chunk["raw_text"]
        )

        self.assertTrue(deadline_chunk["section"].endswith("Good-quality returns"))
        self.assertTrue(shipping_chunk["section"].endswith("Return shipping"))
        self.assertIn("Good-quality returns", deadline_chunk["text"])
        self.assertIn("Return shipping", shipping_chunk["text"])

    def test_long_section_chunks_respect_size_and_preserve_section(self):
        document = {
            "filename": "long.md",
            "metadata": {"title": "Long document"},
            "content": "# Topic\n\n## Details\n\n" + ("A useful sentence with facts. " * 50),
        }

        chunks = chunk_text(document, chunk_size=120, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk["raw_text"]) <= 120 for chunk in chunks))
        self.assertTrue(
            all(chunk["section"].endswith("Details") for chunk in chunks)
        )


if __name__ == "__main__":
    unittest.main()
