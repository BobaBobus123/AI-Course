import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from app.detector import (
    load_model,
    detect,
    DetectionError,
    DEFAULT_CONF_THRESHOLD,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Local Object Detection API — ПР2", lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    """Головна сторінка веб-застосунку для завантаження зображення."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
            "error": None,
            "threshold": DEFAULT_CONF_THRESHOLD,
            "image_data_url": None,
            "filename": None,
        }
    )


@app.post("/api/detect", response_class=JSONResponse)
async def detect_json_endpoint(
    file: Optional[UploadFile] = File(default=None),
    conf_threshold: float = Query(DEFAULT_CONF_THRESHOLD, description="Поріг упевненості від 0 до 1"),
    threshold: Optional[float] = Form(default=None),
):
    """Чистий JSON-ендпоінт детекції об'єктів."""
    effective_threshold = threshold if threshold is not None else conf_threshold

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Файл зображення не передано")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Надісланий файл не є зображенням")

    try:
        image_bytes = await file.read()
        result = detect(image_bytes=image_bytes, conf_threshold=effective_threshold)
        result["filename"] = file.filename
        return JSONResponse(status_code=200, content=result)
    except DetectionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@app.post("/detect")
@app.post("/detect/")
async def detect_universal(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    conf_threshold: float = Query(DEFAULT_CONF_THRESHOLD, description="Поріг упевненості від 0 до 1"),
    threshold: Optional[float] = Form(default=None),
):
    """
    Обробляє і запити з веб-форми браузера (повертає HTML зі знайденими об'єктами),
    і прямі API-запити до /detect/ або /detect (повертає JSON).
    """
    effective_threshold = threshold if threshold is not None else conf_threshold
    accept = request.headers.get("accept", "")
    wants_html = "text/html" in accept

    if file is None or not file.filename:
        msg = "Файл не обрано або не передано"
        if wants_html:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"result": None, "error": msg, "threshold": effective_threshold, "image_data_url": None, "filename": None},
                status_code=400,
            )
        raise HTTPException(status_code=400, detail=msg)

    if file.content_type and not file.content_type.startswith("image/"):
        msg = "Надісланий файл не є зображенням"
        if wants_html:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"result": None, "error": msg, "threshold": effective_threshold, "image_data_url": None, "filename": file.filename},
                status_code=400,
            )
        raise HTTPException(status_code=400, detail=msg)

    try:
        image_bytes = await file.read()
        result = detect(image_bytes=image_bytes, conf_threshold=effective_threshold)
        result["filename"] = file.filename

        if wants_html:
            mime = file.content_type if file.content_type else "image/jpeg"
            b64_img = base64.b64encode(image_bytes).decode("ascii")
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "result": result,
                    "error": None,
                    "threshold": result["threshold"],
                    "image_data_url": f"data:{mime};base64,{b64_img}",
                    "filename": file.filename,
                },
                status_code=200,
            )

        return JSONResponse(status_code=200, content=result)

    except DetectionError as exc:
        if wants_html:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "result": None,
                    "error": exc.message,
                    "threshold": effective_threshold,
                    "image_data_url": None,
                    "filename": file.filename if file else None,
                },
                status_code=exc.status_code,
            )
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)