import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.embeddings import EMBEDDING_BACKEND
from app.index import search_vector
from app.keyword import search_bm25


app = FastAPI(title="Knowledge Base Search API (PR5)")
DEFAULT_THRESHOLD = 0.0 if EMBEDDING_BACKEND == "mock" else 0.5

templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "default_threshold": DEFAULT_THRESHOLD,
            "embedding_backend": EMBEDDING_BACKEND,
        },
    )


def _timed_search(
    search_function: Callable[..., list[dict[str, Any]]], *args: Any, **kwargs: Any
) -> tuple[list[dict[str, Any]], float]:
    start = perf_counter()
    results = search_function(*args, **kwargs)
    return results, round((perf_counter() - start) * 1000, 2)


@app.get("/search")
async def search_endpoint(
    q: str = Query(..., min_length=1, description="Пошуковий запит"),
    top_k: int = Query(5, ge=1, le=100, description="Кількість результатів"),
    threshold: float = Query(
        DEFAULT_THRESHOLD, ge=-1.0, le=1.0, description="Поріг схожості"
    ),
    category: str | None = None,
    audience: str | None = None,
    status: str | None = None,
):
    filters = {
        key: value
        for key, value in {
            "category": category,
            "audience": audience,
            "status": status,
        }.items()
        if value
    }

    try:
        vector_result, keyword_result = await asyncio.gather(
            asyncio.to_thread(
                _timed_search,
                search_vector,
                q,
                top_k=top_k,
                threshold=threshold,
                filters=filters,
            ),
            asyncio.to_thread(
                _timed_search,
                search_bm25,
                q,
                top_k=top_k,
                filters=filters,
            ),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    vector_results, vector_time = vector_result
    bm25_results, bm25_time = keyword_result
    return {
        "query": q,
        "vector_search": {"time_ms": vector_time, "results": vector_results},
        "keyword_search": {"time_ms": bm25_time, "results": bm25_results},
    }
