"""Tool-calling loop for one customer question."""

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from . import llm, tools


load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
TOOL_MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "5"))
if not 1 <= TOOL_MAX_ROUNDS <= 20:
    raise ValueError("TOOL_MAX_ROUNDS must be between 1 and 20")


class ShopUnavailableError(RuntimeError):
    """A shop-service outage occurred while a tool was executing."""

    def __init__(self, payload: dict[str, Any]):
        super().__init__("Сервіс магазину тимчасово недоступний.")
        self.payload = payload


class ShopTimeoutError(TimeoutError):
    """The configured shop-service delay exceeds the tool timeout."""

    def __init__(self, payload: dict[str, Any]):
        super().__init__("Сервіс магазину не відповів у відведений час.")
        self.payload = payload


def _usage_dict(usage: object) -> dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    else:
        data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    return {
        key: int(data.get(key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _remove_unrequested_search_filters(
    tool_name: str, raw_arguments: str, query: str
) -> str:
    if tool_name != "search_catalog":
        return raw_arguments
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return raw_arguments
    if not isinstance(arguments, dict):
        return raw_arguments

    normalized_query = query.casefold()
    for field in ("category", "max_price"):
        value = arguments.get(field)
        if (
            isinstance(value, str)
            and value
            and value.casefold() not in normalized_query
        ):
            arguments.pop(field)
    return json.dumps(arguments, ensure_ascii=False)


def _result(
    answer: str,
    calls: list[dict[str, Any]],
    rounds: int,
    stopped: str,
    model: str,
    elapsed: dict[str, float],
    usage: dict[str, int],
    mode: str,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "calls": calls,
        "rounds": rounds,
        "stopped": stopped,
        "model": model,
        "mode": mode,
        "elapsed": {key: round(value, 4) for key, value in elapsed.items()},
        "usage": usage,
    }


def handle_customer_query(
    customer_id: str, query: str, mode: str | None = None
) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Питання не може бути порожнім.")
    if len(query) > 4000:
        raise ValueError("Питання перевищує дозволений розмір (4000 символів).")
    if not isinstance(customer_id, str) or not customer_id:
        raise ValueError("У сесії не вибрано клієнта.")
    selected_mode = mode or (
        "mock" if llm.MOCK_LLM else llm.DEFAULT_MODE
    )
    if selected_mode not in {"mock", "api", "ollama"}:
        raise ValueError("Режим має бути mock, api або ollama.")

    messages = llm.build_messages(query.strip())
    calls: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    elapsed = {"model": 0.0, "tools": 0.0}
    model_name = llm.MODEL

    for round_number in range(1, TOOL_MAX_ROUNDS + 1):
        response = llm.chat(
            messages, mode=selected_mode, customer_id=customer_id
        )
        elapsed["model"] += response["elapsed"]
        model_name = response["model"]
        current_usage = _usage_dict(response["usage"])
        for key in usage:
            usage[key] += current_usage[key]

        message = response["message"]
        tool_calls = message.tool_calls or []
        if not tool_calls:
            answer = message.content
            if not isinstance(answer, str) or not answer.strip():
                raise llm.LLMResponseError("Модель не повернула текстову відповідь.")
            return _result(
                answer.strip(), calls, round_number, "answer",
                model_name, elapsed, usage, selected_mode,
            )

        messages.append(message.model_dump(exclude_none=True))
        for tool_call in tool_calls:
            call_id = tool_call.id
            name = tool_call.function.name
            raw_arguments = tool_call.function.arguments
            if not isinstance(call_id, str) or not call_id:
                raise llm.LLMResponseError(
                    "Модель повернула виклик інструмента без ідентифікатора."
                )
            if not isinstance(name, str) or not isinstance(raw_arguments, str):
                raise llm.LLMResponseError(
                    "Модель повернула виклик інструмента в неочікуваному форматі."
                )

            raw_arguments = _remove_unrequested_search_filters(
                name, raw_arguments, query
            )
            outcome = tools.call(name, raw_arguments, customer_id)
            elapsed["tools"] += outcome.elapsed
            calls.append(
                {
                    "round": round_number,
                    "name": name,
                    "arguments": outcome.arguments,
                    "raw_arguments": (
                        raw_arguments
                        if len(raw_arguments) <= 8192
                        else "[arguments omitted: over 8192 characters]"
                    ),
                    "status": outcome.status,
                    "reason": outcome.reason,
                    "result": outcome.result,
                    "action": outcome.action,
                    "elapsed": round(outcome.elapsed, 4),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": outcome.content,
                }
            )
            if isinstance(outcome.result, dict):
                code = outcome.result.get("code")
                if code in {"unavailable", "timeout"}:
                    payload = _result(
                        (
                            "Сервіс магазину не відповів у відведений час."
                            if code == "timeout"
                            else "Сервіс магазину тимчасово недоступний. Спробуйте пізніше."
                        ),
                        calls,
                        round_number,
                        f"shop_{code}",
                        model_name,
                        elapsed,
                        usage,
                        selected_mode,
                    )
                    error_type = ShopTimeoutError if code == "timeout" else ShopUnavailableError
                    raise error_type(payload)

    return _result(
        "Не вдалося завершити відповідь у межах дозволеної кількості звертань. "
        "Спробуйте уточнити запит.",
        calls,
        TOOL_MAX_ROUNDS,
        "limit",
        model_name,
        elapsed,
        usage,
        selected_mode,
    )
