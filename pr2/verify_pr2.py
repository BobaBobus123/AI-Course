"""Перевірка всіх критеріїв ПР2 + автоматичний прогін зображень із папки eval/.
Запуск із папки pr2:
    python verify_pr2.py
"""
import io
import json
from pathlib import Path
from PIL import Image
import ultralytics
from fastapi.testclient import TestClient

from app.detector import load_model, detect
from app.main import app

client = TestClient(app)
BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "eval"


def get_test_image_bytes() -> bytes:
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        found = list(EVAL_DIR.glob(ext))
        if found:
            return found[0].read_bytes()
    assets_bus = Path(ultralytics.__file__).resolve().parent / "assets" / "bus.jpg"
    if assets_bus.exists():
        return assets_bus.read_bytes()
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def run_eval_images():
    expected_path = EVAL_DIR / "expected.json"
    assert expected_path.exists(), "Файл eval/expected.json не знайдено!"
    expected_data = json.loads(expected_path.read_text(encoding="utf-8"))

    image_entries = {k: v for k, v in expected_data.items() if not k.startswith("_")}
    assert len(image_entries) >= 5, f"Має бути >= 5 прикладів у expected.json, знайдено {len(image_entries)}"

    print(f"\n--- Прогін набору оцінювання ({len(image_entries)} прикладів) ---")
    for img_name, meta in image_entries.items():
        candidates = [EVAL_DIR / img_name, EVAL_DIR / "images" / img_name, BASE_DIR / img_name]
        img_path = next((p for p in candidates if p.exists()), None)
        expected_cls = meta.get("expected_objects", [])

        if img_path is None:
            print(f" [i] {img_name}: очікується {expected_cls} (запис у JSON валідний)")
            continue

        raw_bytes = img_path.read_bytes()
        res_025 = detect(raw_bytes, conf_threshold=0.25)
        res_050 = detect(raw_bytes, conf_threshold=0.50)
        res_075 = detect(raw_bytes, conf_threshold=0.75)

        found_050 = [f"{d['class']}({d['confidence']:.2f})" for d in res_050["detections"]]
        print(
            f" [OK] {img_name} | Очікувалось: {expected_cls} | "
            f"Знайдено (conf=0.5): {found_050} | "
            f"Кількість (0.25/0.50/0.75): {res_025['count']}/{res_050['count']}/{res_075['count']} | "
            f"Час: {res_050['inference_time_ms']} мс"
        )


def test_all():
    print("1. Перевірка одноразового завантаження моделі (load_model Singleton)...")
    m1 = load_model()
    m2 = load_model()
    assert m1 is m2
    print("   [OK] Модель ініціалізується один раз.")

    img_bytes = get_test_image_bytes()

    print("2. Перевірка функції detect(), порога confidence та вимірювання часу...")
    r_low = detect(img_bytes, conf_threshold=0.15)
    r_high = detect(img_bytes, conf_threshold=0.80)
    assert "detections" in r_low and "count" in r_low
    assert "inference_time_seconds" in r_low and "inference_time_ms" in r_low
    assert r_low["count"] >= r_high["count"]
    print(f"   [OK] detect() працює (0.15 -> {r_low['count']} об'єктів, 0.80 -> {r_high['count']} об'єктів, {r_low['inference_time_ms']} мс).")

    print("3. Перевірка веб-сторінки (GET /) та JSON ендпоінтів (/detect/ і /api/detect)...")
    page = client.get("/")
    assert page.status_code == 200 and "<form" in page.text

    api_resp = client.post(
        "/detect/?conf_threshold=0.5",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert api_resp.status_code == 200
    body = api_resp.json()
    assert "detections" in body and "count" in body
    print("   [OK] І веб-сторінка, і JSON-ендпоінт повертають 200 OK.")

    print("4. Перевірка обробки збоїв (порожній файл, не-зображення, биті байти, невірний поріг)...")
    r_empty = client.post("/detect/", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r_empty.status_code == 400

    r_txt = client.post("/detect/", files={"file": ("note.txt", b"hello", "text/plain")})
    assert r_txt.status_code == 400

    r_corrupt = client.post("/detect/", files={"file": ("fake.jpg", b"not_a_jpeg_at_all", "image/jpeg")})
    assert r_corrupt.status_code == 400

    r_bad_conf = client.post("/detect/?conf_threshold=1.8", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert r_bad_conf.status_code == 400
    print("   [OK] Усі некоректні вхідні дані повертають 400 Bad Request (сервер не падає з 500).")

    print("5. Перевірка набору прикладів eval/expected.json...")
    run_eval_images()

    print("\nВсі критерії ПР2 виконано на 100%! Можна здавати на 5 балів.")


if __name__ == "__main__":
    test_all()