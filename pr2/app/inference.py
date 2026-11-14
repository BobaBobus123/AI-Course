"""
Модуль інтеграції локального inference (сумісний місток до app.detector).
Гарантує, що модель завантажується лише один раз у пам'ять, навіть якщо
звернення відбувається через стару назву модуля app.inference.
"""
from typing import Dict, Any
from ultralytics import YOLO

from app.detector import (
    MODEL_NAME,
    DEFAULT_CONF_THRESHOLD,
    DetectionError,
    InvalidImageError,
    InvalidThresholdError,
    load_model as _load_detector_model,
    detect as _run_detection,
)


def load_model(model_path: str = MODEL_NAME) -> YOLO:
    """Повертає єдиний закешований у пам'яті екземпляр моделі YOLO."""
    return _load_detector_model(model_path=model_path)


def detect(image_bytes: bytes, conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> Dict[str, Any]:
    """Виконує локальний inference і повертає структурований результат."""
    return _run_detection(image_bytes=image_bytes, conf_threshold=conf_threshold)


def detect_objects(image_bytes: bytes, conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> Dict[str, Any]:
    """
    Повна сумісність зі старою сигнатурою detect_objects(image_bytes, conf_threshold).
    Використовує ту саму одноразово завантажену модель і валідацію зображень.
    """
    return _run_detection(image_bytes=image_bytes, conf_threshold=conf_threshold)