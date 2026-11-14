"""FastAPI endpoints for the customer assistant demo."""

import html
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.sessions import SessionMiddleware

from shop import service
from . import assistant, llm, tools


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
logger = logging.getLogger(__name__)

app = FastAPI(title="Помічник клієнта — ПР8")
configured_session_secret = os.getenv("SESSION_SECRET")
if not configured_session_secret:
    logger.warning(
        "SESSION_SECRET is unset; using an ephemeral development-only session key."
    )
session_secret = configured_session_secret or secrets.token_urlsafe(32)
app.add_middleware(
    SessionMiddleware,
    secret_key=session_secret,
    same_site="lax",
    https_only=os.getenv("SESSION_HTTPS_ONLY", "false").strip().lower()
    in {"1", "true", "yes", "on"},
)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["mock", "api", "ollama"] | None = None


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(pattern=r"^C-[0-9]{4,12}$")


def _page(error: str | None = None) -> str:
    document = INDEX_PAGE.read_text(encoding="utf-8")
    schema_json = json.dumps(tools.specs(), ensure_ascii=False).replace("<", "\\u003c")
    embedded = f'<script id="tools-schema" type="application/json">{schema_json}</script>'
    if error is not None:
        panel = (
            '<section class="server-result error"><h2>Помилка</h2><pre>'
            f"{html.escape(error)}</pre></section>"
        )
        document = document.replace("<h1>", panel + "<h1>", 1)
    return document.replace("<script>", embedded + "<script>", 1)


def _error_response(error: Exception) -> JSONResponse:
    if isinstance(error, assistant.ShopUnavailableError):
        return JSONResponse(
            status_code=503,
            content={
                "detail": str(error),
                **error.payload,
            },
        )
    if isinstance(error, assistant.ShopTimeoutError):
        return JSONResponse(
            status_code=504,
            content={"detail": str(error), **error.payload},
        )
    if isinstance(error, llm.LLMTimeoutError):
        return JSONResponse(status_code=504, content={"detail": str(error)})
    if isinstance(error, llm.LLMProviderTimeoutError):
        return JSONResponse(status_code=504, content={"detail": str(error)})
    if isinstance(error, (llm.LLMRateLimitError, llm.LLMUnavailableError)):
        return JSONResponse(status_code=503, content={"detail": str(error)})
    if isinstance(error, llm.LLMConfigurationError):
        return JSONResponse(status_code=503, content={"detail": str(error)})
    if isinstance(error, llm.LLMServiceError):
        return JSONResponse(status_code=502, content={"detail": str(error)})
    if isinstance(error, ValueError):
        return JSONResponse(status_code=400, content={"detail": str(error)})
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутрішня помилка застосунку. Спробуйте пізніше."},
    )


def _known_customer(customer_id: str) -> bool:
    return any(
        customer.get("customer_id") == customer_id
        for customer in service.list_customers()
    )


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return _page()


@app.get("/api/customers")
def api_customers() -> list[dict]:
    """Demo login choices; production must replace this with authenticated identity."""
    return service.list_customers()


@app.get("/api/session")
def api_get_session(request: Request) -> dict:
    customer_id = request.session.get("customer_id")
    if not isinstance(customer_id, str) or not _known_customer(customer_id):
        request.session.pop("customer_id", None)
        return {"customer_id": None}
    return {"customer_id": customer_id}


@app.post("/api/session")
def api_set_session(payload: SessionRequest, request: Request) -> dict:
    if not _known_customer(payload.customer_id):
        raise HTTPException(status_code=404, detail="Клієнта не знайдено.")
    request.session.clear()
    request.session["customer_id"] = payload.customer_id
    return {"customer_id": payload.customer_id}


@app.get("/api/tools")
def api_tools() -> list[dict]:
    return tools.specs()


@app.get("/api/config")
def api_config() -> dict:
    return {
        "default_mode": llm.DEFAULT_MODE,
        "api_key_configured": bool(llm.API_KEY),
        "ollama_model": llm.OLLAMA_MODEL,
    }


@app.post("/api/reset")
def api_reset() -> dict:
    service.reset()
    return {"ok": True}


@app.post("/api/ask")
def api_ask(payload: AskRequest, request: Request) -> dict:
    customer_id = request.session.get("customer_id")
    if not isinstance(customer_id, str) or not _known_customer(customer_id):
        request.session.pop("customer_id", None)
        raise HTTPException(
            status_code=401,
            detail="Виберіть клієнта в демонстраційній сесії перед запитом.",
        )
    try:
        return assistant.handle_customer_query(
            customer_id, payload.question, payload.mode
        )
    except Exception as error:
        if isinstance(
            error,
            (
                llm.LLMServiceError,
                ValueError,
                assistant.ShopUnavailableError,
                assistant.ShopTimeoutError,
            ),
        ):
            return _error_response(error)
        logger.exception("Unexpected assistant request failure")
        return _error_response(error)


@app.post("/ask", response_class=HTMLResponse)
async def ask_form(request: Request) -> HTMLResponse:
    form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
    question = form.get("query", form.get("question", [""]))[0]
    requested_mode = form.get("mode", [None])[0]
    mode = requested_mode if requested_mode in {"mock", "api", "ollama"} else None
    customer_id = request.session.get("customer_id")
    if not isinstance(customer_id, str) or not _known_customer(customer_id):
        return HTMLResponse(
            _page(error="Виберіть клієнта в демонстраційній сесії перед запитом."),
            status_code=401,
        )
    try:
        result = assistant.handle_customer_query(customer_id, question, mode)
        document = _page()
        answer_panel = (
            '<section class="server-result"><h2>Відповідь</h2><pre>'
            f"{html.escape(result['answer'])}</pre></section>"
        )
        return HTMLResponse(document.replace("<h1>", answer_panel + "<h1>", 1))
    except Exception as error:
        if not isinstance(
            error,
            (
                llm.LLMServiceError,
                ValueError,
                assistant.ShopUnavailableError,
                assistant.ShopTimeoutError,
            ),
        ):
            logger.exception("Unexpected assistant form failure")
        response = _error_response(error)
        detail = response.body.decode("utf-8", errors="replace")
        try:
            message = json.loads(detail).get("detail", "Помилка запиту.")
        except json.JSONDecodeError:
            message = "Помилка запиту."
        return HTMLResponse(_page(error=message), status_code=response.status_code)
