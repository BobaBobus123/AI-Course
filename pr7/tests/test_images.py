import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image

from app import images


class PrepareImageTests(unittest.TestCase):
    @staticmethod
    def png_bytes(mode="RGBA"):
        buffer = io.BytesIO()
        color = (255, 0, 0, 128) if mode == "RGBA" else (255, 0, 0)
        Image.new(mode, (20, 10), color).save(buffer, format="PNG")
        return buffer.getvalue()

    def test_transparent_png_is_encoded_as_jpeg(self):
        data_url, metadata = images.prepare_image(self.png_bytes())
        payload = base64.b64decode(data_url.split(",", 1)[1])
        with Image.open(io.BytesIO(payload)) as result:
            self.assertEqual(result.format, "JPEG")
            self.assertEqual(result.mode, "RGB")
        self.assertEqual(metadata["original_size"], (20, 10))
        self.assertGreater(metadata["sent_bytes"], 0)

    def test_palette_png_with_transparency_is_encoded_as_jpeg(self):
        buffer = io.BytesIO()
        image = Image.new("P", (20, 10), 0)
        image.info["transparency"] = 0
        image.save(buffer, format="PNG")

        data_url, _ = images.prepare_image(buffer.getvalue())

        with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as result:
            self.assertEqual(result.mode, "RGB")
            self.assertEqual(result.format, "JPEG")

    def test_oversized_upload_is_rejected_before_decode(self):
        with patch.object(images, "MAX_UPLOAD_BYTES", 4):
            with self.assertRaises(images.ImageTooLargeError):
                images.prepare_image(b"12345")

    def test_invalid_file_is_rejected(self):
        with self.assertRaises(images.ImageInputError):
            images.prepare_image(b"not an image")

    def test_exif_orientation_is_applied(self):
        buffer = io.BytesIO()
        image = Image.new("RGB", (20, 10), "white")
        exif = Image.Exif()
        exif[274] = 6
        image.save(buffer, format="JPEG", exif=exif)

        _, metadata = images.prepare_image(buffer.getvalue())

        self.assertEqual(metadata["original_size"], (20, 10))
        self.assertEqual(metadata["new_size"], (10, 20))


if __name__ == "__main__":
    unittest.main()
