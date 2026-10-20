import html
import json
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from shop import service
from . import assistant, tools

app = FastAPI(title="Помічник клієнта — ПР8")
INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"

class AskRequest(BaseModel):
    customer_id: str
    question: str

def _page(result: dict | None = None, error: str | None = None) -> str:
    document = INDEX_PAGE.read_text(encoding="utf-8")
    schema_json = json.dumps(tools.TOOLS_SCHEMA, ensure_ascii=False).replace("<", "\\u003c")
    embedded = f'<script id="tools-schema" type="application/json">{schema_json}</script>'
    if result is not None:
        content = html.escape(result.get("answer", ""))
        panel = f'<section class="server-result"><h2>Відповідь</h2><pre>{content}</pre></section>'
        document = document.replace("<h1>", panel + "<h1>", 1)
    elif error is not None:
        panel = f'<section class="server-result error"><h2>Помилка</h2><pre>{html.escape(error)}</pre></section>'
        document = document.replace("<h1>", panel + "<h1>", 1)
    return document.replace("<script>", embedded + "<script>", 1)

@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return _page()

@app.get("/api/customers")
def api_customers() -> list[dict]:
    return service.list_customers()

@app.get("/api/tools")
def api_tools() -> list[dict]:
    return tools.TOOLS_SCHEMA

@app.post("/api/reset")
def api_reset() -> dict:
    service.reset()
    return {"ok": True}

@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    try:
        return assistant.handle_customer_query(payload.customer_id, payload.question)
    except ValueError as error:
        return JSONResponse(status_code=400, content={"detail": str(error)})
    except Exception as error:
        return JSONResponse(status_code=502, content={"detail": f"Не вдалося отримати відповідь: {error}"})

@app.post("/ask", response_class=HTMLResponse)
async def ask_form(request: Request) -> HTMLResponse:
    form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
    customer_id = form.get("customer_id", [""])[0]
    query = form.get("query", form.get("question", [""]))[0]
    try:
        result = assistant.handle_customer_query(customer_id, query)
        return HTMLResponse(_page(result=result))
    except ValueError as error:
        return HTMLResponse(_page(error=str(error)), status_code=400)
    except Exception as error:
        return HTMLResponse(_page(error=f"Не вдалося отримати відповідь: {error}"), status_code=502)