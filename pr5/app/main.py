import time
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional
from app.index import search_vector
from app.keyword import search_bm25

app = FastAPI(title="Knowledge Base Search API (PR5)")

templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/search")
async def search_endpoint(
    q: str = Query(..., description="Пошуковий запит"),
    top_k: int = Query(5, description="Кількість результатів"),
    threshold: float = Query(0.5, description="Поріг схожості"),
    category: Optional[str] = None,
    audience: Optional[str] = None,
    status: Optional[str] = None
):
    filters = {}
    if category: filters["категорія"] = category
    if audience: filters["аудиторія"] = audience
    if status: filters["статус"] = status

    # Семантичний пошук
    start_time = time.time()
    vector_results = search_vector(q, top_k=top_k, threshold=threshold, filters=filters)
    vector_time = round((time.time() - start_time) * 1000, 2)

    # Пошук за ключовими словами (BM25)
    start_time = time.time()
    bm25_results = search_bm25(q, top_k=top_k, filters=filters)
    bm25_time = round((time.time() - start_time) * 1000, 2)

    return {
        "query": q,
        "vector_search": {
            "time_ms": vector_time,
            "results": vector_results
        },
        "keyword_search": {
            "time_ms": bm25_time,
            "results": bm25_results
        }
    }