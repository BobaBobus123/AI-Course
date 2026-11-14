"""Автоматична самоперевірка всіх критеріїв приймання ПР1.
Запуск із папки pr1:
    python verify_pr1.py
"""
from unittest.mock import patch, MagicMock
import requests
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_all():
    print("1. Перевірка реальних міст (JSON та HTML)...")
    for city in ("Київ", "Кременчук"):
        r_json = client.get("/api/weather", params={"city": city})
        assert r_json.status_code == 200, f"Очікувався 200, отримано {r_json.status_code}"
        body = r_json.json()
        assert body["status"] == "ok"
        assert "temperature" in body["data"] and "wind_speed" in body["data"]

        r_html = client.get("/weather", params={"city": city})
        assert r_html.status_code == 200
        assert "Температура" in r_html.text
    print("   [OK] Реальні міста працюють бездоганно.")

    print("2. Перевірка порожнього рядка (має бути 400)...")
    r_empty = client.get("/api/weather", params={"city": "   "})
    assert r_empty.status_code == 400
    assert r_empty.json()["error"]["type"] == "bad_request"
    print("   [OK] Порожній ввід повертає 400.")

    print("3. Перевірка неіснуючого міста (немає поля 'results' -> 404)...")
    r_not_found = client.get("/api/weather", params={"city": "Абвгдежз123456789"})
    assert r_not_found.status_code == 404
    assert r_not_found.json()["error"]["type"] == "not_found"
    print("   [OK] Неіснуюче місто повертає 404.")

    print("4. Перевірка таймауту (має бути 504, сервер не падає)...")
    with patch("app.weather.requests.get", side_effect=requests.exceptions.Timeout):
        r_timeout = client.get("/api/weather", params={"city": "Київ"})
        assert r_timeout.status_code == 504
        assert r_timeout.json()["error"]["type"] == "timeout"
    print("   [OK] Таймаут обробляється коректно (504).")

    print("5. Перевірка різної реакції на 4xx та 5xx від зовнішнього API...")
    mock_500 = MagicMock(status_code=503)
    with patch("app.weather.requests.get", return_value=mock_500):
        r_5xx = client.get("/api/weather", params={"city": "Київ"})
        assert r_5xx.status_code == 503
        assert r_5xx.json()["error"]["type"] == "upstream_server_error"

    mock_400 = MagicMock(status_code=400)
    with patch("app.weather.requests.get", return_value=mock_400):
        r_4xx = client.get("/api/weather", params={"city": "Київ"})
        assert r_4xx.status_code == 400
        assert r_4xx.json()["error"]["type"] == "upstream_client_error"
    print("   [OK] Коди 4xx і 5xx розрізняються коректно.")

    print("6. Перевірка несподіваного формату JSON (відсутні поля -> 502)...")
    mock_bad_json = MagicMock(status_code=200)
    mock_bad_json.json.return_value = {"results": [{"name": "Київ"}]}  # немає latitude/longitude
    with patch("app.weather.requests.get", return_value=mock_bad_json):
        r_bad = client.get("/api/weather", params={"city": "Київ"})
        assert r_bad.status_code == 502
        assert r_bad.json()["error"]["type"] == "missing_fields"
    print("   [OK] Неповний JSON перехоплюється без падіння (502).")

    print("\nВсі 6 перевірок ПР1 пройдено на 100%! Можна здавати на 5 балів.")


if __name__ == "__main__":
    test_all()