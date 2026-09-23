import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI, APIError
from fastapi import HTTPException
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-3.6-flash")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

if not API_KEY:
    raise ValueError(f"API ключ не знайдено! Перевірте файл .env за шляхом: {env_path}")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Структура виводу за допомогою Pydantic (Structured Output) ---
class SupportResponseSchema(BaseModel):
    response_text: str = Field(description="Ввічлива та чітка відповідь клієнту українською мовою.")
    topic: Literal["доставка", "оплата", "повернення", "гарантія", "інше"] = Field(description="Тема звернення з фіксованого переліку.")
    is_based_on_rules: bool = Field(description="True, якщо відповідь повністю базується на правилах з context.md; False, якщо правила про це мовчать або питання поза їх межами.")
    needs_clarification: bool = Field(description="True, якщо запит двозначний і вимагає уточнення деталей від клієнта.")
    escalate_to_operator: bool = Field(description="True, якщо розмову необхідно передати живому оператору.")
    order_number: Optional[str] = Field(default=None, description="Номер замовлення у форматі ORD-XXXXX, якщо клієнт згадав його в діалозі.")

def load_context() -> str:
    context_path = os.path.join(os.path.dirname(__file__), "..", "context.md")
    if os.path.exists(context_path):
        with open(context_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Правила не знайдено."

def generate_dialog_response(
    history: List[dict], 
    user_query: str, 
    temperature: float = DEFAULT_TEMPERATURE,
    mode: str = "context_engineering"
) -> dict:
    context_text = load_context() if mode == "context_engineering" else ""
    
    system_instruction = (
        "Ти — професійний асистент служби підтримки інтернет-магазину. "
        "Відповідай на звернення клієнта ВИКЛЮЧНО на підставі наданих правил магазину (якщо вони передані). "
        "Якщо відповіді немає в правилах, чесно скажи про це і нічого не вигадуй. "
        "Завжди враховуй історію розмови та попередні повідомлення користувача."
    )

    if mode == "fuzzy_query":
        messages = [{"role": "user", "content": user_query}]
    elif mode == "structured_prompt":
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Звернення клієнта: {user_query}"}
        ]
    else:  # context_engineering з історією та правилами
        messages = [{"role": "system", "content": f"{system_instruction}\n\nПравила магазину:\n{context_text}"}]
        
        # Бюджет історії (останні 6 реплік для збереження контексту)
        limited_history = history[-6:] if history else []
        for msg in limited_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        messages.append({"role": "user", "content": user_query})

    start_time = time.time()
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS,
            response_format=SupportResponseSchema,
            timeout=20.0
        )
        inference_time = time.time() - start_time
        parsed_result = completion.choices[0].message.parsed
        
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        return {
            "parsed_data": parsed_result.model_dump(),
            "model": MODEL_NAME,
            "execution_time_seconds": round(inference_time, 4),
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

    except Exception as e:
        logger.error("Помилка генерації або валідації: %s", e)
        raise HTTPException(
            status_code=502, 
            detail=f"Помилка обробки запиту моделлю або валідації схеми: {str(e)}"
        )