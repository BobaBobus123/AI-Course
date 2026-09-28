import time
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from app.retrieval import retrieve_and_filter_chunks
from app.llm import generate_rag_response

app = FastAPI(title="RAG Support Assistant (PR6)")

templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/ask")
async def ask_endpoint(q: str = Query(..., description="Питання клієнта")):
    print(f"Отримано запит на сервер: {q}")  # Додайте цей рядок для перевірки
    ...
    chunks = retrieve_and_filter_chunks(q)

    # 1. Пошук та відбір фрагментів (з урахуванням захисту коду)
    start_search = time.time()
    chunks = retrieve_and_filter_chunks(q)
    search_time = round((time.time() - start_search) * 1000, 2)

    # 2. Перевірка: чи є фрагменти вище порога
    if not chunks:
        return {
            "query": q,
            "search_time_ms": search_time,
            "generation_time_seconds": 0.0,
            "result": {
                "response_text": "Вибачте, у нашій базі знань немає інформації за вашим запитом.",
                "found_in_base": False,
                "sources": [],
                "raw_chunks": [],
                "model": "none",
                "tokens": {"total_tokens": 0}
            }
        }

    # 3. Генерація відповіді через LLM із контекстом
    try:
        rag_result = generate_rag_response(q, chunks)
    except Exception as e:
        return {"error": str(e)}

    return {
        "query": q,
        "search_time_ms": search_time,
        "generation_time_seconds": rag_result["generation_time_seconds"],
        "result": rag_result
    }