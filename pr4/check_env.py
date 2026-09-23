import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-3.6-flash")

print(f"[*] Перевірка конфігурації для ПР4...")
print(f" - Шлях до .env: {env_path} (існує: {env_path.exists()})")
print(f" - Модель: {MODEL_NAME}")
print(f" - API Key знайдено: {bool(API_KEY)}")

if not API_KEY:
    print("[X] Помилка: API ключ не знайдено у файлі .env!")
    exit(1)

class TestSchema(BaseModel):
    status: str = Field(description="Статус перевірки")
    code: int = Field(description="Числовий код")

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Поверни статус 'OK' та код 200"}],
        response_format=TestSchema
    )
    print(f"[OK] З'єднання з Gemini API та Structured Output працюють успішно!")
    print(f" - Відповідь моделі: {completion.choices[0].message.parsed}")
except Exception as e:
    print(f"[X] Помилка підключення або підтримки Structured Output: {e}")