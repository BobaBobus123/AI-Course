import requests
from typing import Dict, Any, List
from datetime import datetime


class WeatherIntegrationError(Exception):
    """Базовий виняток для помилок інтеграції з погодним API."""
    def __init__(self, message: str, status_code: int = 500, error_type: str = "integration_error"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)


class InvalidCityQueryError(WeatherIntegrationError):
    """Помилка некоректного вводу (порожній рядок тощо) -> 400."""
    def __init__(self, message: str = "Назва міста не може бути порожньою."):
        super().__init__(message, status_code=400, error_type="bad_request")


class CityNotFoundError(WeatherIntegrationError):
    """Місто не знайдено у геокодингу -> 404."""
    def __init__(self, city: str):
        super().__init__(
            f"Місто '{city}' не знайдено. Перевірте правильність написання.",
            status_code=404,
            error_type="not_found"
        )


def get_weather_description(code: int, is_day: int = 1) -> tuple[str, str]:
    descriptions = {
        0: ("Ясно", "☀️" if is_day else "🌙"),
        1: ("Переважно ясно", "🌤️" if is_day else "🌙"),
        2: ("Мінлива хмарність", "⛅"),
        3: ("Хмарно", "☁️"),
        45: ("Туман", "🌫️"),
        48: ("Туман", "🌫️"),
        51: ("Мряка", "🌧️"),
        53: ("Помірна мряка", "🌧️"),
        55: ("Густа мряка", "🌧️"),
        61: ("Слабкий дощ", "🌧️"),
        63: ("Помірний дощ", "🌧️"),
        65: ("Сильний дощ", "🌧️"),
        71: ("Слабкий сніг", "❄️"),
        73: ("Помірний сніг", "❄️"),
        75: ("Сильний сніг", "❄️"),
        80: ("Зливовий дощ", "🌦️"),
        95: ("Гроза", "⛈️"),
    }
    return descriptions.get(code, ("Невідомо", "🌡️"))


def _handle_http_status(resp: requests.Response, service_name: str) -> None:
    """Різна реакція на коди помилок клієнта (4xx) та сервера (5xx)."""
    if 400 <= resp.status_code < 500:
        raise WeatherIntegrationError(
            f"Помилка запиту до сервісу {service_name} (код {resp.status_code}). Перевірте параметри.",
            status_code=400,
            error_type="upstream_client_error"
        )
    if resp.status_code >= 500:
        raise WeatherIntegrationError(
            f"Сервер {service_name} тимчасово недоступний (код {resp.status_code}). Спробуйте пізніше.",
            status_code=503,
            error_type="upstream_server_error"
        )


def parse_geocoding_response(geo_data: Any, queried_city: str) -> Dict[str, Any]:
    """Розбирає JSON геокодування та перевіряє наявність обов'язкових полів."""
    if not isinstance(geo_data, dict):
        raise WeatherIntegrationError("Несподіваний формат відповіді від геокодування.", 502, "bad_format")

    # Для незнайденого міста Open-Meteo не повертає поле "results" взагалі
    results = geo_data.get("results")
    if not results or not isinstance(results, list):
        raise CityNotFoundError(queried_city)

    first_match = results[0] if isinstance(results[0], dict) else {}
    lat = first_match.get("latitude")
    lon = first_match.get("longitude")
    resolved_name = first_match.get("name") or queried_city
    country = first_match.get("country", "")

    if lat is None or lon is None:
        raise WeatherIntegrationError("У відповіді геокодування відсутні координати міста.", 502, "missing_fields")

    return {
        "name": str(resolved_name),
        "country": str(country),
        "latitude": float(lat),
        "longitude": float(lon),
    }


def parse_weather_response(weather_data: Any, location_info: Dict[str, Any]) -> Dict[str, Any]:
    """Розбирає JSON прогнозу погоди та перевіряє наявність температури і швидкості вітру."""
    if not isinstance(weather_data, dict):
        raise WeatherIntegrationError("Несподіваний формат даних від погодного сервісу.", 502, "bad_format")

    # Підтримка і блоку 'current', і блоку 'current_weather' (як у samples)
    current = weather_data.get("current") or weather_data.get("current_weather")
    if not isinstance(current, dict):
        raise WeatherIntegrationError("Відсутні дані про поточну погоду у відповіді сервера.", 502, "missing_fields")

    temperature = current.get("temperature_2m", current.get("temperature"))
    windspeed = current.get("wind_speed_10m", current.get("windspeed"))
    weathercode = current.get("weather_code", current.get("weathercode", 0))
    is_day = current.get("is_day", 1)

    if temperature is None or windspeed is None:
        raise WeatherIntegrationError("У відповіді сервісу відсутні показники температури або вітру.", 502, "missing_fields")

    desc, icon = get_weather_description(
        int(weathercode) if weathercode is not None else 0,
        int(is_day) if is_day is not None else 1
    )

    daily_forecasts = []
    daily_data = weather_data.get("daily")
    if isinstance(daily_data, dict):
        times = daily_data.get("time") or []
        t_max = daily_data.get("temperature_2m_max") or []
        t_min = daily_data.get("temperature_2m_min") or []
        w_codes = daily_data.get("weathercode", daily_data.get("weather_code")) or []

        days_count = min(len(times), len(t_max), len(t_min), len(w_codes))
        for i in range(days_count):
            if t_max[i] is None or t_min[i] is None:
                continue
            day_code = int(w_codes[i]) if w_codes[i] is not None else 0
            _, day_icon = get_weather_description(day_code, 1)
            try:
                date_obj = datetime.strptime(str(times[i]), "%Y-%m-%d")
                formatted_date = date_obj.strftime("%d.%m")
            except ValueError:
                formatted_date = str(times[i])

            daily_forecasts.append({
                "date": formatted_date,
                "temp_max": round(float(t_max[i])),
                "temp_min": round(float(t_min[i])),
                "icon": day_icon
            })

    return {
        "city": location_info["name"],
        "country": location_info.get("country", ""),
        "latitude": location_info["latitude"],
        "longitude": location_info["longitude"],
        "temperature": float(temperature),
        "wind_speed": float(windspeed),
        "windspeed": float(windspeed),  # дублюємо ключ для сумісності з шаблоном index.html
        "description": desc,
        "icon": icon,
        "daily": daily_forecasts
    }


def get_weather(city_name: str) -> Dict[str, Any]:
    cleaned_city = (city_name or "").strip()
    if not cleaned_city:
        raise InvalidCityQueryError("Введіть коректну назву міста (рядок не може бути порожнім).")

    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    weather_url = "https://api.open-meteo.com/v1/forecast"

    try:
        # 1. Запит до API геокодування з параметрами та timeout
        geo_params = {
            "name": cleaned_city,
            "count": 1,
            "language": "uk",
            "format": "json"
        }
        geo_resp = requests.get(geo_url, params=geo_params, timeout=5.0)
        _handle_http_status(geo_resp, "геокодування")

        try:
            geo_data = geo_resp.json()
        except ValueError:
            raise WeatherIntegrationError("Сервіс геокодування повернув некоректний JSON.", 502, "bad_json")

        location_info = parse_geocoding_response(geo_data, cleaned_city)

        # 2. Запит до API прогнозу погоди з параметрами (рядки через кому) та timeout
        weather_params = {
            "latitude": location_info["latitude"],
            "longitude": location_info["longitude"],
            "current": "temperature_2m,wind_speed_10m,weather_code,is_day",
            "current_weather": "true",
            "daily": "temperature_2m_max,temperature_2m_min,weathercode",
            "timezone": "auto"
        }
        weather_resp = requests.get(weather_url, params=weather_params, timeout=5.0)
        _handle_http_status(weather_resp, "прогнозу погоди")

        try:
            weather_data = weather_resp.json()
        except ValueError:
            raise WeatherIntegrationError("Погодний сервіс повернув некоректний JSON.", 502, "bad_json")

        return parse_weather_response(weather_data, location_info)

    except WeatherIntegrationError:
        raise
    except requests.exceptions.Timeout:
        raise WeatherIntegrationError(
            "Перевищено час очікування відповіді від зовнішнього сервісу (timeout).",
            status_code=504,
            error_type="timeout"
        )
    except requests.exceptions.ConnectionError:
        raise WeatherIntegrationError(
            "Зовнішній погодний сервіс недоступний. Перевірте підключення до мережі.",
            status_code=503,
            error_type="network_unavailable"
        )
    except requests.exceptions.RequestException:
        raise WeatherIntegrationError(
            "Помилка під час мережевого звернення до зовнішнього API.",
            status_code=502,
            error_type="request_error"
        )
    except (KeyError, TypeError, ValueError):
        raise WeatherIntegrationError(
            "Отримано несподіваний формат даних від зовнішнього сервісу.",
            status_code=502,
            error_type="parse_error"
        )


def get_city_suggestions(query: str) -> List[Dict[str, str]]:
    cleaned = (query or "").strip()
    if len(cleaned) < 2:
        return []
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": cleaned, "count": 5, "language": "uk", "format": "json"}
    try:
        resp = requests.get(geo_url, params=geo_params, timeout=3.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if not isinstance(data, dict):
            return []
        suggestions = []
        for r in data.get("results") or []:
            if not isinstance(r, dict) or not r.get("name"):
                continue
            name = r["name"]
            country = r.get("country", "")
            suggestions.append({
                "name": name,
                "label": f"{name}, {country}" if country else name
            })
        return suggestions
    except (requests.RequestException, ValueError):
        return []