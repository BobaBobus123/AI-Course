from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.weather import get_weather, WeatherIntegrationError, get_city_suggestions

app = FastAPI(title="Weather App ПР1")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"weather": None, "error": None}
    )

@app.get("/weather", response_class=HTMLResponse)
def get_weather_ui(request: Request, city: str = Query(..., min_length=1)):
    try:
        weather_data = get_weather(city)
        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={"city": weather_data["city"], "weather": weather_data, "error": None}
        )
    except WeatherIntegrationError as e:
        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={"city": city, "weather": None, "error": e.message},
            status_code=e.status_code
        )

@app.get("/api/suggest")
def suggest_cities(query: str = Query(..., min_length=2)):
    """API для отримання списку міст при введенні тексту."""
    return get_city_suggestions(query)