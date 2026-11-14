import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app import llm
from app.images import prepare_image
from app.main import app
from fastapi.testclient import TestClient


class MockLLMTests(unittest.TestCase):
    def test_mock_extractor_matches_uploaded_training_fixture(self):
        project_dir = Path(__file__).resolve().parents[1]
        sample_path = project_dir / "samples" / "clean" / "rahunok-02.png"
        data_url, _ = prepare_image(sample_path.read_bytes())

        with (
            patch.object(llm, "MOCK_LLM", True),
            patch.object(llm, "_create_client") as create_client,
        ):
            invoice, metadata = llm.extract_invoice_data(data_url)

        create_client.assert_not_called()
        self.assertEqual(invoice.number, "58")
        self.assertEqual(invoice.supplier.name, "ФОП Коваленко Ірина Петрівна")
        self.assertEqual(invoice.total_to_pay, "12095.00")
        self.assertTrue(metadata["mock"])
        self.assertEqual(metadata["mock_sample"], "clean/rahunok-02.png")
        self.assertEqual(metadata["total_tokens"], 0)

    def test_api_returns_data_for_matching_fixture(self):
        project_dir = Path(__file__).resolve().parents[1]
        sample_path = project_dir / "samples" / "clean" / "rahunok-02.png"
        expected = json.loads(
            (project_dir / "samples" / "expected.json").read_text(encoding="utf-8")
        )["clean/rahunok-02.png"]["fields"]

        with patch.object(llm, "MOCK_LLM", True):
            response = TestClient(app).post(
                "/api/process",
                files={
                    "file": (
                        sample_path.name,
                        sample_path.read_bytes(),
                        "image/png",
                    )
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["decision"], "auto")
        self.assertEqual(payload["data"]["number"], expected["number"])
        self.assertEqual(
            payload["data"]["supplier"]["name"], expected["supplier"]["name"]
        )
        self.assertEqual(payload["meta"]["mock_sample"], "clean/rahunok-02.png")
        self.assertTrue(payload["meta"]["mock"])
        self.assertIn("mockNotice", TestClient(app).get("/").text)

    def test_unknown_image_does_not_receive_a_fabricated_invoice(self):
        buffer = io.BytesIO()
        Image.new("RGB", (20, 10), "white").save(buffer, format="PNG")

        with patch.object(llm, "MOCK_LLM", True):
            response = TestClient(app).post(
                "/api/process",
                files={"file": ("unlisted.png", buffer.getvalue(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"]["number"], None)
        self.assertIsNone(payload["meta"]["mock_sample"])
        self.assertIn("не збігається", payload["meta"]["mock_warning"])


if __name__ == "__main__":
    unittest.main()
