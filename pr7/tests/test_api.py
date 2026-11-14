import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.llm import LLMRateLimitError, LLMResponseError, LLMTimeoutError
from app.main import app


class UploadApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_empty_file_returns_400(self):
        response = self.client.post(
            "/api/process",
            files={"file": ("empty.png", b"", "image/png")},
        )
        self.assertEqual(response.status_code, 400)

    def test_oversized_file_returns_413(self):
        with patch.object(main, "MAX_UPLOAD_BYTES", 4):
            response = self.client.post(
                "/api/process",
                files={"file": ("large.png", b"12345", "image/png")},
            )
        self.assertEqual(response.status_code, 413)

    def test_non_image_returns_400(self):
        response = self.client.post(
            "/api/process",
            files={"file": ("not-image.png", b"not an image", "image/png")},
        )
        self.assertEqual(response.status_code, 400)

    @staticmethod
    def png_file():
        import io
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
        return buffer.getvalue()

    def test_model_timeouts_and_failures_return_gateway_statuses(self):
        test_cases = (
            (LLMTimeoutError("timeout"), 504),
            (LLMRateLimitError("quota"), 503),
            (LLMResponseError("invalid response"), 502),
        )
        for error, expected_status in test_cases:
            with self.subTest(expected_status=expected_status):
                with patch.object(main, "process", side_effect=error):
                    response = self.client.post(
                        "/api/process",
                        files={"file": ("invoice.png", self.png_file(), "image/png")},
                    )
                self.assertEqual(response.status_code, expected_status)
                self.assertIn("detail", response.json())


if __name__ == "__main__":
    unittest.main()
