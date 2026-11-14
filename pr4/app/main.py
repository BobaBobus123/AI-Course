from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app.inference import (
    DEFAULT_TEMPERATURE,
    TOKEN_BUDGET,
    generate_dialog_response,
)
from app.schemas import get_json_schema_dict

app = FastAPI(
    title="TechStore Context-Aware Support Assistant API (PR4)",
    description=(
        "Діалоговий AI-асистент підтримки з керуванням контекстом, "
        "бюджетом токенів та Structured Output"
    ),
    version="1.0.0",
)

conversation_history: List[Dict[str, str]] = []
INDEX_TEMPLATE = Path(__file__).resolve().parent / "templates" / "index.html"


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Повідомлення клієнта")
    temperature: Optional[float] = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=1.5)
    mode: Literal["fuzzy_query", "structured_prompt", "context_engineering"] = Field(
        default="context_engineering",
        description="Рівень керування контекстом",
    )
    token_budget: Optional[int] = Field(
        default=TOKEN_BUDGET,
        ge=300,
        le=16000,
        description="Бюджет токенів на вхідний контекст",
    )

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Повідомлення не може бути порожнім.")
        return normalized


@app.get("/schema")
async def get_schema_endpoint():
    """Повертає JSON Schema відповіді помічника для перевірки."""
    return get_json_schema_dict()


@app.get("/history")
async def get_history_endpoint():
    """Повертає поточну серверну історію діалогу."""
    return {
        "messages_count": len(conversation_history),
        "history": conversation_history,
    }


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    clean_query = payload.query.strip()
    result = generate_dialog_response(
        history=conversation_history,
        user_query=clean_query,
        temperature=(
            payload.temperature
            if payload.temperature is not None
            else DEFAULT_TEMPERATURE
        ),
        mode=payload.mode,
        token_budget=(
            payload.token_budget
            if payload.token_budget is not None
            else TOKEN_BUDGET
        ),
    )

    if payload.mode == "context_engineering":
        assistant_text = result["response_text"]
        conversation_history.extend(
            [
                {"role": "user", "content": clean_query},
                {"role": "assistant", "content": assistant_text},
            ]
        )

    return result


@app.post("/reset")
async def reset_history():
    conversation_history.clear()
    return {"status": "success", "message": "Історію діалогу очищено."}


@app.get("/", response_class=FileResponse)
async def chat_page():
    return FileResponse(INDEX_TEMPLATE, media_type="text/html; charset=utf-8")
