from pathlib import Path
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.weather import get_weather, WeatherIntegrationError, get_city_suggestions

app = FastAPI(title="Weather App ПР1")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    """Головна сторінка застосунку."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"city": "", "weather": None, "error": None}
    )


@app.get("/weather", response_class=HTMLResponse)
def get_weather_ui(request: Request, city: str = Query(default="")):
    """HTML-сторінка з результатами погоди або зрозумілим повідомленням про помилку."""
    try:
        weather_data = get_weather(city)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "city": weather_data["city"],
                "weather": weather_data,
                "error": None
            },
            status_code=200
        )
    except WeatherIntegrationError as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "city": city,
                "weather": None,
                "error": e.message
            },
            status_code=e.status_code
        )


@app.get("/api/weather", response_class=JSONResponse)
def get_weather_api(city: str = Query(default="")):
    """
    JSON-ендпоінт (вимога критеріїв приймання ПР1):
    повертає погоду як JSON (200 OK) або структуровану помилку з відповідним HTTP-статусом.
    """
    try:
        weather_data = get_weather(city)
        return JSONResponse(status_code=200, content={"status": "ok", "data": weather_data})
    except WeatherIntegrationError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "status": "error",
                "error": {
                    "type": e.error_type,
                    "message": e.message
                }
            }
        )


@app.get("/api/suggest", response_class=JSONResponse)
def suggest_cities(query: str = Query(default="")):
    """API для автодоповнення назв міст."""
    return get_city_suggestions(query)