import time
from ultralytics import YOLO
from PIL import Image
import io

model = YOLO("yolov8n.pt")

def detect_objects(image_bytes: bytes, conf_threshold: float = 0.5) -> dict:
    image = Image.open(io.BytesIO(image_bytes))

    start_time = time.time()
    results = model.predict(image, conf=conf_threshold)
    inference_time = time.time() - start_time

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[box.cls[0].item()],
                "box": box.xyxy[0].tolist(),
                "confidence": round(float(box.conf[0].item()), 4)
            })

    return {
        "detections": detections,
        "count": len(detections),
        "inference_time_seconds": round(inference_time, 4)
    }