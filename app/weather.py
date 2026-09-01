import requests
from typing import Dict, Any, List
from datetime import datetime

class WeatherIntegrationError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class CityNotFoundError(WeatherIntegrationError):
    def __init__(self, city: str):
        super().__init__(f"Місто '{city}' не знайдено. Перевірте правильність написання.", 404)

def get_weather_description(code: int, is_day: int) -> tuple[str, str]:
    descriptions = {
        0: ("Ясно", "☀️" if is_day else "🌙"),
        1: ("Переважно ясно", "🌤️" if is_day else "🌙"),
        2: ("Мінлива хмарність", "⛅"),
        3: ("Хмарно", "☁️"),
        45: ("Туман", "🌫️"),
        48: ("Туман", "🌫️"),
        51: ("Мряка", "🌧️"),
        61: ("Слабкий дощ", "🌧️"),
        63: ("Помірний дощ", "🌧️"),
        65: ("Сильний дощ", "🌧️"),
        71: ("Слабкий сніг", "❄️"),
        73: ("Помірний сніг", "❄️"),
        75: ("Сильний сніг", "❄️"),
        95: ("Гроза", "⛈️"),
    }
    return descriptions.get(code, ("Невідомо", "🌡️"))

def get_weather(city_name: str) -> Dict[str, Any]:
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    weather_url = "https://api.open-meteo.com/v1/forecast"
    
    try:
        geo_params = {"name": city_name, "count": 1, "language": "uk", "format": "json"}
        geo_resp = requests.get(geo_url, params=geo_params, timeout=5.0)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        
        if "results" not in geo_data or not geo_data["results"]:
            raise CityNotFoundError(city_name)
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        resolved_name = geo_data["results"][0].get("name", city_name)
        
        # Додаємо параметри для денного прогнозу
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "daily": ["temperature_2m_max", "temperature_2m_min", "weathercode"],
            "timezone": "auto"
        }
        weather_resp = requests.get(weather_url, params=weather_params, timeout=5.0)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        
        if "current_weather" not in weather_data:
            raise WeatherIntegrationError("Відсутні дані про погоду у відповіді сервера.", 502)
            
        current = weather_data["current_weather"]
        desc, icon = get_weather_description(current.get("weathercode", -1), current.get("is_day", 1))
        
        # Формуємо список прогнозів на кожен день
        daily_forecasts = []
        if "daily" in weather_data:
            daily_data = weather_data["daily"]
            for i in range(len(daily_data["time"])):
                day_code = daily_data["weathercode"][i]
                # Для денного прогнозу вважаємо, що це день (1)
                day_desc, day_icon = get_weather_description(day_code, 1)
                
                # Форматуємо дату (з "2026-09-01" у "01.09")
                date_obj = datetime.strptime(daily_data["time"][i], "%Y-%m-%d")
                formatted_date = date_obj.strftime("%d.%m")
                
                daily_forecasts.append({
                    "date": formatted_date,
                    "temp_max": round(daily_data["temperature_2m_max"][i]),
                    "temp_min": round(daily_data["temperature_2m_min"][i]),
                    "icon": day_icon
                })
        
        return {
            "city": resolved_name,
            "temperature": current.get("temperature"),
            "windspeed": current.get("windspeed"),
            "description": desc,
            "icon": icon,
            "daily": daily_forecasts
        }
        
    except requests.exceptions.Timeout:
        raise WeatherIntegrationError("Перевищено час очікування відповіді від погодного сервісу.", 504)
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        raise WeatherIntegrationError(f"Помилка зовнішнього API ({status}).", 502)
    except requests.exceptions.RequestException:
        raise WeatherIntegrationError("Проблема з мережевим з'єднанням.", 503)
    except ValueError:
        raise WeatherIntegrationError("Отримано некоректний формат даних від сервісу.", 502)

def get_city_suggestions(query: str) -> List[Dict[str, str]]:
    if len(query) < 2:
        return []
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": query, "count": 5, "language": "uk", "format": "json"}
    try:
        resp = requests.get(geo_url, params=geo_params, timeout=3.0)
        resp.raise_for_status()
        data = resp.json()
        suggestions = []
        for r in data.get("results", []):
            name = r.get("name")
            country = r.get("country", "")
            suggestions.append({
                "name": name,
                "label": f"{name}, {country}" if country else name
            })
        return suggestions
    except Exception:
        return []