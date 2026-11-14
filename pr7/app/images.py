import base64
import io
import os
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

UPLOAD_MAX_MB = float(os.getenv("UPLOAD_MAX_MB", "10"))
MAX_UPLOAD_BYTES = int(UPLOAD_MAX_MB * 1024 * 1024)
MAX_IMAGE_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))
MAX_IMAGE_PIXELS = int(os.getenv("IMAGE_MAX_PIXELS", "40000000"))

if UPLOAD_MAX_MB <= 0 or MAX_IMAGE_SIDE <= 0 or MAX_IMAGE_PIXELS <= 0:
    raise ValueError("Image size and upload limits must be positive")

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class ImageInputError(ValueError):
    """The uploaded content is not a supported, valid image."""


class ImageTooLargeError(ImageInputError):
    """The upload exceeds a configured byte or pixel limit."""


def prepare_image(image_bytes: bytes) -> tuple[str, dict[str, Any]]:
    if not image_bytes:
        raise ImageInputError("Завантажений файл порожній.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ImageTooLargeError(
            f"Файл перевищує дозволений розмір {UPLOAD_MAX_MB:g} МБ."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image_stream = io.BytesIO(image_bytes)
            with Image.open(image_stream) as probe:
                if probe.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ImageInputError(
                        "Підтримуються лише файли JPEG, PNG та WEBP."
                    )
                original_size = probe.size
                if original_size[0] * original_size[1] > MAX_IMAGE_PIXELS:
                    raise ImageTooLargeError(
                        "Зображення перевищує дозволену кількість пікселів."
                    )
                probe.verify()

            image_stream.seek(0)
            with Image.open(image_stream) as opened:
                image = ImageOps.exif_transpose(opened)
                image.load()

            if max(image.size) > MAX_IMAGE_SIDE:
                image.thumbnail(
                    (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
                    Image.Resampling.LANCZOS,
                )

            if (
                image.mode in ("RGBA", "LA")
                or (image.mode == "P" and "transparency" in image.info)
            ):
                image = image.convert("RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")

            sent_size = image.size
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85, optimize=True)
    except ImageInputError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeError(
            "Зображення перевищує безпечний ліміт пікселів."
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageInputError(
            "Файл не є коректним зображенням або пошкоджений."
        ) from exc

    data_url = "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")
    return data_url, {
        "original_size": original_size,
        "new_size": sent_size,
        "original_bytes": len(image_bytes),
        "sent_bytes": buffer.tell(),
    }
