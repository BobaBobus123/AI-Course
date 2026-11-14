from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from app.extraction import process
from app.images import MAX_UPLOAD_BYTES, ImageInputError, ImageTooLargeError
from app.llm import (
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServiceError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.rules import ReferenceDataError


app = FastAPI(title="Мультимодальний аналіз рахунків (ПР7)")
templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/process")
async def process_document(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Завантажений файл порожній.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Файл перевищує дозволений розмір "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} МБ."
            ),
        )

    try:
        result = await run_in_threadpool(process, content)
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ImageInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (LLMRateLimitError, LLMUnavailableError, LLMConfigurationError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (LLMResponseError, LLMServiceError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ReferenceDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    image = result.image
    return {
        "decision": result.decision,
        "reasons": result.reasons,
        "issues": result.issues,
        "data": result.document,
        "meta": {
            "image_size_orig": image.get("original_size"),
            "image_size_new": image.get("new_size"),
            "image_bytes_orig": image.get("original_bytes"),
            "image_bytes_sent": image.get("sent_bytes"),
            "time_prep": result.elapsed.get("prepare"),
            "time_llm": result.elapsed.get("extraction"),
            "time_rules": result.elapsed.get("checks"),
            "time_total": round(sum(result.elapsed.values()), 3),
            "tokens": {
                key: value
                for key, value in (result.usage or {}).items()
                if key.endswith("_tokens")
            },
            "model": result.model,
            "mock": bool(result.usage and result.usage.get("mock")),
            "mock_sample": (
                result.usage.get("mock_sample") if result.usage else None
            ),
            "mock_warning": (
                result.usage.get("mock_warning") if result.usage else None
            ),
        },
    }
