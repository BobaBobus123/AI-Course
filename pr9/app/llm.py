import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from . import tools

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.getenv("LLM_MODEL_NAME", "gemini-3.8-flash")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60.0)

SYSTEM_PROMPT = """Ти — помічник клієнта магазину "Сузір'я". 
Викликай інструменти для отримання реальних даних. Ніколи не вигадуй статуси чи ціни."""

def build_messages(query: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

def chat_with_model(messages: list) -> dict:
    start_time = time.perf_counter()
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools.TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.0
            )
            return {
                "message": response.choices[0].message,
                "elapsed": time.perf_counter() - start_time,
                "model": MODEL,
                "usage": response.usage
            }
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 3:
                time.sleep(5)
                continue
            raise e