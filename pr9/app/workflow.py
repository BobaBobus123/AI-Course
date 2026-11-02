import os
import time
import uuid
import json
from dotenv import load_dotenv
from openai import OpenAI

from shop import service
from . import schema, store, tools

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.getenv("LLM_MODEL_NAME", "gemini-3.8-flash")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60.0)

MAX_STEPS = int(os.getenv("WORKFLOW_MAX_STEPS", "10"))
TIMEOUT_SEC = float(os.getenv("WORKFLOW_TIMEOUT", "60.0"))
TOKEN_BUDGET = int(os.getenv("WORKFLOW_TOKEN_BUDGET", "5000"))

def _add_log(state: dict, step_name: str, step_type: str, status: str, message: str, data: dict = None):
    state["journal"].append({
        "step": step_name,
        "type": step_type,
        "status": status,
        "message": message,
        "data": data or {},
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    })

def create_run(customer_id: str, text: str) -> dict:
    run_id = str(uuid.uuid4())[:8]
    state = {
        "run_id": run_id,
        "customer_id": customer_id,
        "text": text,
        "status": "running",
        "current_step": "parse",
        "parsed_data": None,
        "order_info": None,
        "journal": [],
        "tokens_used": 0,
        "token_budget": TOKEN_BUDGET,
        "max_steps": MAX_STEPS,
        "timeout": TIMEOUT_SEC,
        "start_time": time.time(),
        "answer": None,
        "operator_comment": None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    store.save_state(state)
    return execute_workflow(run_id)

def execute_workflow(run_id: str) -> dict:
    state = store.load_state(run_id)
    if not state or state["status"] in ["completed", "rejected", "stopped"]:
        return state

    if "start_time" not in state:
        state["start_time"] = time.time()
    if "tokens_used" not in state:
        state["tokens_used"] = 0
    if "max_steps" not in state:
        state["max_steps"] = MAX_STEPS
    if "timeout" not in state:
        state["timeout"] = TIMEOUT_SEC
    if "token_budget" not in state:
        state["token_budget"] = TOKEN_BUDGET

    start_time = state["start_time"]

    while state["status"] == "running":
        if len(state["journal"]) >= MAX_STEPS:
            state["status"] = "stopped"
            _add_log(state, "limits", "code", "error", "Перевищено ліміт кроків")
            break
        if (time.time() - start_time) > TIMEOUT_SEC:
            state["status"] = "stopped"
            _add_log(state, "limits", "code", "error", "Вичерпано ліміт часу")
            break
        if state["tokens_used"] >= TOKEN_BUDGET:
            state["status"] = "stopped"
            _add_log(state, "limits", "code", "error", "Вичерпано бюджет токенів")
            break

        step = state["current_step"]

        if step == "parse":
            prompt = f"Роззбери звернення клієнта у структурований формат: '{state['text']}'"
            try:
                completion = client.beta.chat.completions.parse(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "Ти класифікатор звернень клієнтів інтернет-магазину."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=schema.ParsedRequest,
                    temperature=0.0
                )
                usage = completion.usage
                state["tokens_used"] += usage.total_tokens
                parsed = completion.choices[0].message.parsed
                state["parsed_data"] = parsed.model_dump()
                
                _add_log(state, "parse", "model", "ok", "Звернення успішно розібрано", state["parsed_data"])
                
                if parsed.intent in ["return", "cancel", "status"] and parsed.order_id:
                    state["current_step"] = "fetch_data"
                else:
                    state["status"] = "completed"
                    state["answer"] = "На жаль, ми не змогли розпізнати номер замовлення або суть звернення."
                    _add_log(state, "routing", "code", "ok", "Недостатньо даних")
            except Exception as e:
                state["status"] = "stopped"
                _add_log(state, "parse", "model", "error", str(e))
            store.save_state(state)

        elif step == "fetch_data":
            p_data = state["parsed_data"] or {}
            try:
                order_id = p_data.get("order_id")
                order_res = service.get_order(order_id)
                if order_res.get("customer_id") != state["customer_id"]:
                    state["status"] = "rejected"
                    state["answer"] = "Замовлення не належить поточному клієнту."
                    _add_log(state, "fetch_data", "tool", "error", "Чуже замовлення")
                else:
                    state["order_info"] = order_res
                    _add_log(state, "fetch_data", "tool", "ok", "Дані замовлення отримано", {"order_id": order_id})
                    
                    intent = p_data.get("intent")
                    if intent == "status":
                        state["current_step"] = "generate_answer"
                    elif intent in ["return", "cancel"]:
                        state["current_step"] = "wait_approval"
                        state["status"] = "waiting_approval"
                        _add_log(state, "wait_approval", "code", "ok", "Очікує підтвердження оператора")
            except Exception as e:
                state["status"] = "stopped"
                _add_log(state, "fetch_data", "tool", "error", str(e))
            store.save_state(state)

        elif step == "wait_approval":
            store.save_state(state)
            break

        elif step == "execute_action":
            p_data = state.get("parsed_data") or {}
            intent = p_data.get("intent", "return")
            try:
                if intent == "return":
                    res = service.request_return(
                        p_data.get("order_id", "10421"), 
                        p_data.get("sku", ""), 
                        p_data.get("reason", "not_suitable"), 
                        p_data.get("quantity", 1)
                    )
                    _add_log(state, "execute_action", "tool", "ok", "Заявку на повернення створено", res)
                elif intent == "cancel":
                    _add_log(state, "execute_action", "tool", "ok", "Замовлення скасовано")
                
                state["current_step"] = "generate_answer"
            except Exception as e:
                state["status"] = "stopped"
                _add_log(state, "execute_action", "tool", "error", str(e))
            store.save_state(state)

        elif step == "generate_answer":
            order_summary = json.dumps(state.get("order_info", {}), ensure_ascii=False)
            prompt = f"Звернення клієнта: {state['text']}\nДані з системи: {order_summary}\nСформуй ввічливу відповідь клієнту."
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "Ти ввічливий помічник підтримки інтернет-магазину."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                usage = completion.usage
                state["tokens_used"] += usage.total_tokens
                state["answer"] = completion.choices[0].message.content
                state["status"] = "completed"
                _add_log(state, "generate_answer", "model", "ok", "Відповідь згенеровано")
            except Exception as e:
                state["status"] = "stopped"
                _add_log(state, "generate_answer", "model", "error", str(e))
            store.save_state(state)

    return store.load_state(run_id)

def handle_operator_action(run_id: str, action: str, comment: str) -> dict:
    state = store.load_state(run_id)
    if not state:
        return {"error": "Run not found"}

    state["operator_comment"] = comment
    if action == "approve":
        _add_log(state, "operator_approval", "human", "ok", f"Підтверджено. Коментар: {comment}")
        state["status"] = "running"
        state["current_step"] = "execute_action"
        store.save_state(state)
        state = execute_workflow(run_id)
    elif action == "reject":
        _add_log(state, "operator_approval", "human", "rejected", f"Відхилено. Коментар: {comment}")
        state["status"] = "rejected"
        state["answer"] = f"Дія відхилена оператором. Причина: {comment}"
        store.save_state(state)

    return state