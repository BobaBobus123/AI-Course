import os
import time
import logging
from pathlib import Path
from openai import OpenAI, APIError, APITimeoutError, RateLimitError, AuthenticationError
from fastapi import HTTPException
from dotenv import load_dotenv

# Явно вказуємо шлях до файлу .env у папці pr3
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))

if not API_KEY:
    raise ValueError(f"API ключ не знайдено! Перевірте наявність файлу .env за шляхом: {env_path}")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_context() -> str:
    context_path = os.path.join(os.path.dirname(__file__), "..", "context.md")
    if os.path.exists(context_path):
        with open(context_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Правила не знайдено."

def generate_support_response(user_query: str, temperature: float = DEFAULT_TEMPERATURE) -> dict:
    context_text = load_context()
    
    system_instruction = (
        "Ти — професійний асистент служби підтримки інтернет-магазину[cite: 3]. "
        "Відповідай на звернення клієнта ВИКЛЮЧНО на підставі наданих правил магазину (контексту)[cite: 3]. "
        "Якщо відповіді немає в правилах, прямо скажи про це і нічого не вигадуй[cite: 3]. "
        "Якщо звернення можна зрозуміти по-різному, уточни деталі[cite: 3]. "
        "Пиши чітко, ввічливо та зрозуміло для клієнта."
    )

    user_prompt = f"Контекст (правила магазину):\n{context_text}\n\nЗвернення клієнта: {user_query}"

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS,
            timeout=15.0
        )
        inference_time = time.time() - start_time
        answer = response.choices[0].message.content

        return {
            "response": answer,
            "model": MODEL_NAME,
            "execution_time_seconds": round(inference_time, 4)
        }

    except RateLimitError as e:
        logger.error("Перевищено ліміт запитів (429): %s", e)
        raise HTTPException(
            status_code=429, 
            detail="Перевищено ліміт запитів до мовної моделі (429). Будь ласка, зачекайте хвилину і спробуйте знову[cite: 3]."
        )
    except AuthenticationError as e:
        logger.error("Помилка автентифікації (401): %s", e)
        raise HTTPException(
            status_code=401, 
            detail="Помилка автентифікації API ключа (401). Перевірте правильність налаштувань[cite: 3]."
        )
    except APITimeoutError as e:
        logger.error("Сплив час очікування (Timeout): %s", e)
        raise HTTPException(
            status_code=504, 
            detail="Мовна модель відповідає надто довго (таймаут). Спробуйте пізніше[cite: 3]."
        )
    except APIError as e:
        logger.error("Помилка API провайдера (5xx): %s", e)
        raise HTTPException(
            status_code=502, 
            detail=f"Тимчасова недоступність сервісу моделі (5xx): {str(e)}[cite: 3]"
        )
    except Exception as e:
        logger.error("Непередбачена помилка: %s", e)
        raise HTTPException(
            status_code=500, 
            detail=f"Внутрішня помилка сервера: {str(e)}"
        )