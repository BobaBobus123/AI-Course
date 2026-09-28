import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")

print(f"[*] Перевірка конфігурації для ПР6...")
print(f" - Шлях до .env: {env_path} (існує: {env_path.exists()})")
print(f" - LLM Модель: {MODEL_NAME}")
print(f" - Модель ембедінгів: {EMBEDDING_MODEL}")
print(f" - API Key знайдено: {bool(API_KEY)}")

if not API_KEY:
    print("[X] Помилка: API ключ не знайдено у файлі .env!")
    exit(1)

try:
    print("[*] Завантаження моделі ембедінгів для перевірки...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    test_vec = model.encode(["Тестовий запит для RAG"])
    print(f"[OK] Ембедінг успішно створено. Розмірність: {len(test_vec[0])}")
except Exception as e:
    print(f"[X] Помилка моделі ембедінгів: {e}")
    exit(1)

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Привіт! Відпиши одним словом: 'Працює'"}],
        max_tokens=10
    )
    answer = completion.choices[0].message.content.strip()
    print(f"[OK] З'єднання з LLM API успішне! Відповідь моделі: {answer}")
except Exception as e:
    print(f"[X] Помилка підключення до LLM API: {e}")