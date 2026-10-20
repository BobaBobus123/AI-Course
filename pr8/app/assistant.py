import json
import time
from dotenv import load_dotenv
from . import llm, tools

load_dotenv()
TOOL_MAX_ROUNDS = 5

def _usage_dict(usage: object) -> dict:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }

def handle_customer_query(customer_id: str, query: str) -> dict:
    if not query.strip():
        raise ValueError("Питання не може бути порожнім")

    messages = llm.build_messages(query)
    calls = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    elapsed = {"model": 0.0, "tools": 0.0}
    model_name = getattr(llm, "MODEL", "gemini-3.8-flash")
    rounds = 0

    while rounds < TOOL_MAX_ROUNDS:
        rounds += 1
        response = llm.chat_with_model(messages)
        elapsed["model"] += response["elapsed"]
        model_name = response["model"]
        current_usage = _usage_dict(response["usage"])
        for key in usage:
            usage[key] += current_usage.get(key, 0) or 0

        message = response["message"]
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return {
                "answer": message.content or "Модель не повернула текстову відповідь.",
                "calls": calls,
                "rounds": rounds,
                "stopped": "answer",
                "model": model_name,
                "elapsed": elapsed,
                "usage": usage,
            }

        messages.append(message.model_dump(exclude_none=True))
        for tool_call in tool_calls:
            name = tool_call.function.name
            raw_arguments = tool_call.function.arguments or "{}"
            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Аргументи мають бути JSON-об'єктом")
                before = time.perf_counter()
                content = tools.execute_tool(name, arguments, customer_id)
                tool_elapsed = time.perf_counter() - before
                parsed = json.loads(content)
                has_error = isinstance(parsed, dict) and "error" in parsed
                code = parsed.get("code") if isinstance(parsed, dict) else None
                status = "rejected" if code in ("invalid_arguments", "unknown_tool") else "error" if has_error else "ok"
                reason = parsed.get("error") if has_error else None
            except (json.JSONDecodeError, ValueError) as error:
                tool_elapsed = 0.0
                content = json.dumps({"error": f"Некоректний JSON аргументів: {error}", "code": "invalid_arguments"}, ensure_ascii=False)
                arguments = raw_arguments
                status = "rejected"
                reason = "Модель повернула некоректні аргументи"

            elapsed["tools"] += tool_elapsed
            calls.append({
                "round": rounds,
                "name": name,
                "arguments": raw_arguments,
                "status": status,
                "reason": reason,
                "result": content,
                "elapsed": tool_elapsed,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": content,
            })

    return {
        "answer": "Не вдалося завершити відповідь у межах ліміту викликів моделі. Спробуйте уточнити запит.",
        "calls": calls,
        "rounds": rounds,
        "stopped": "limit",
        "model": model_name,
        "elapsed": elapsed,
        "usage": usage,
    }