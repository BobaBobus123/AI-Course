from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from app.llm import LLMRateLimitError, LLMServiceError, LLMTimeoutError
from app.rag import run_rag


app = FastAPI(title="RAG Support Assistant (PR6)")

templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/ask")
async def ask_endpoint(
    q: str = Query(..., min_length=1, max_length=2000, description="Питання клієнта"),
):
    try:
        return await run_in_threadpool(run_rag, q.strip())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Індекс бази знань не знайдено. Спочатку запустіть "
                "python ingest.py у папці pr6."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except LLMRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
