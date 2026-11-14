import io
import time
from typing import Dict, Any, List, Optional
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
DEFAULT_CONF_THRESHOLD = 0.5

_model_instance: Optional[YOLO] = None


class DetectionError(Exception):
    """Базовый класс исключений модуля детекции."""
    def __init__(self, message: str, status_code: int = 400, error_type: str = "detection_error"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)


class InvalidImageError(DetectionError):
    """Ошибка некорректного или пустого файла изображения -> 400."""
    def __init__(self, message: str = "Надісланий файл не є коректним зображенням або порожній."):
        super().__init__(message, status_code=400, error_type="invalid_image")


class InvalidThresholdError(DetectionError):
    """Ошибка некорректного порога confidence -> 400."""
    def __init__(self, message: str = "Поріг упевненості (conf_threshold) має бути від 0.0 до 1.0."):
        super().__init__(message, status_code=400, error_type="invalid_threshold")


def load_model(model_path: str = MODEL_NAME) -> YOLO:
    """
    Загружает pretrained-модель YOLO один раз в память (Singleton)
    и переиспользует её для всех последующих запросов.
    """
    global _model_instance
    if _model_instance is None:
        try:
            _model_instance = YOLO(model_path)
        except Exception as exc:
            raise DetectionError(
                f"Не вдалося ініціалізувати модель '{model_path}': {exc}",
                status_code=500,
                error_type="model_load_error"
            )
    return _model_instance


def _open_and_validate_image(image_bytes: bytes) -> Image.Image:
    """Проверяет, что байты не пустые и являются валидным изображением."""
    if not image_bytes:
        raise InvalidImageError("Файл порожній")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise InvalidImageError("Надісланий файл не є зображенням або пошкоджений")


def detect(image_bytes: bytes, conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> Dict[str, Any]:
    """
    Выполняет локальный inference на изображении, фильтрует объекты по порогу confidence,
    измеряет время выполнения и возвращает структурированный результат.
    """
    try:
        threshold = float(conf_threshold)
    except (TypeError, ValueError):
        raise InvalidThresholdError()

    if not (0.0 <= threshold <= 1.0):
        raise InvalidThresholdError()

    image = _open_and_validate_image(image_bytes)
    width, height = image.size
    model = load_model()

    start_time = time.perf_counter()
    try:
        results = model.predict(source=image, conf=0.05, verbose=False)
    except Exception as exc:
        raise DetectionError(
            f"Помилка під час виконання inference: {exc}",
            status_code=500,
            error_type="inference_error"
        )
    inference_sec = time.perf_counter() - start_time
    inference_ms = round(inference_sec * 1000.0, 2)

    detections: List[Dict[str, Any]] = []
    filtered_out = 0

    for result in results:
        names = result.names or {}
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf_val = round(float(box.conf[0].item()), 4)
            cls_id = int(box.cls[0].item())
            cls_name = str(names.get(cls_id, cls_id))
            coords = [round(float(x), 1) for x in box.xyxy[0].tolist()]

            if conf_val >= threshold:
                detections.append({
                    "class": cls_name,
                    "label": cls_name,
                    "class_id": cls_id,
                    "box": coords,
                    "bbox": coords,
                    "confidence": conf_val,
                })
            else:
                filtered_out += 1

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "model": MODEL_NAME,
        "image_size": {"width": width, "height": height},
        "detections": detections,
        "objects": detections,
        "count": len(detections),
        "filtered_out_count": filtered_out,
        "threshold": round(threshold, 2),
        "inference_time_seconds": round(inference_sec, 4),
        "inference_time_ms": inference_ms,
    }


def detect_objects(image_bytes: bytes, conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> Dict[str, Any]:
    """Алиас для обратной совместимости со старым кодом."""
    return detect(image_bytes=image_bytes, conf_threshold=conf_threshold)