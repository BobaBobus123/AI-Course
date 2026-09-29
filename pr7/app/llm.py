import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from app.schema import InvoiceSchema

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """Ти — суворий фінансовий аналізатор. Твоє завдання — вилучити дані з документа у форматі JSON.
КРИТИЧНІ ПРАВИЛА:
1. ПЕРЕПИСУЙ, А НЕ ВИПРАВЛЯЙ. Якщо в документі є арифметична помилка, пиши цифри точно так, як вони надруковані.
2. ВІДСУТНЄ = NULL. Якщо дати, номера чи суми немає, або документ обрізано і підсумку не видно — повертай null. Не вираховуй підсумки самостійно!
3. Формат сум: лише цифри та крапка (наприклад, "12570.00", "150.50"). Без пробілів.
4. Текст на кшталт "погоджено", "оплачено" — ігноруй, це не частина реквізитів рахунку."""

def extract_invoice_data(data_url: str, max_retries: int = 4) -> tuple[InvoiceSchema, dict]:
    """Відправляє зображення та інструкцію до LLM. Робить автоматичні повтори при помилці 503."""
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Опрацюй цей документ та поверни JSON за заданою схемою."},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                temperature=0.0,
                response_format=InvoiceSchema
            )
            
            end_time = time.time()
            result = response.choices[0].message.parsed
            usage = response.usage
            
            meta = {
                "generation_time_sec": round(end_time - start_time, 2),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
            
            return result, meta
            
        except Exception as e:
            error_str = str(e)
            # Якщо це помилка перевантаження (503) або ліміту запитів (429)
            if ("503" in error_str or "429" in error_str) and attempt < max_retries - 1:
                print(f"Тимчасова помилка API. Спроба {attempt + 1} з {max_retries}. Очікування 20 секунд...")
                time.sleep(20) # Збільшуємо паузу, щоб ліміт 429 точно встиг оновитися
                continue
            
            # Якщо це інша помилка або спроби вичерпано - прокидаємо далі
            raise e