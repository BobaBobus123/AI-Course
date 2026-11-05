"""Кеш відповідей: однакове питання — одна оплата.

Повтори беруться звідусіль: «Повторити» без правок, пʼять натискань
поспіль, двоє операторів над одним зверненням. Кеш повертає готовий
результат, не звертаючись до провайдера, — і це видно операторові,
журналу й бюджету.

Модуль знає `Request` і `Result` з контракту провайдера і більше
нічого: ані хто питає, ані навіщо. Шлюз (`app/gateway.py`) перед
першою спробою питає `get(make_key(...))`, після вдалої генерації
кладе `put`; прапорець `fresh` у `Request.metadata` каже кеш оминути.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* що входить у ключ: усе, що впливає на відповідь (провайдер, модель,
  версія prompt, повідомлення, температура, `max_tokens`), і нічого з
  того, що не впливає (клієнт, номер чернетки, час). Канонічне подання
  — `json.dumps(..., sort_keys=True)` → `hashlib.sha256`;
* термін життя: факти звернення змінюються; що таке «застаріло» для
  чернетки — година, день? Коли видаляти прострочене — при читанні чи
  окремо;
* сховище: словник у памʼяті (зникає з перезапуском) чи файл JSON у
  `state/` (переживає, але потребує обмеження розміру й атомарного
  запису). Що робити, коли записів більше за `MAX_ENTRIES`;
* що кешувати: результат запасного варіанту? підставного провайдера?
  відповідь, у якій `finish_reason` не `stop`;
* `fresh` оминає читання — а запис? Нова чернетка після «Повторити»
  має замінити стару в кеші чи ні;
* пʼять однакових запитів одночасно промахнуться всі — чи має другий
  чекати на перший (обʼєднання запитів у польоті) і скільки;
* лічильники для `stats`: влучання, промахи, обходи, записів зараз;
  чи скидати їх із перезапуском.
"""

import hashlib
import json
import os
import tempfile
import threading
import time

from dotenv import load_dotenv

from .providers.base import Request, Result

load_dotenv()

ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
TTL_SECONDS = float(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))
# Порожнє значення — кеш лише в памʼяті процесу.
CACHE_FILE = os.getenv("CACHE_FILE", "state/cache.json")
_lock = threading.RLock()
_entries: dict[str, dict] = {}
_loaded = False
_hits = 0
_misses = 0
_bypassed = 0


def settings() -> dict:
    """Чинні налаштування — для сторінки й журналу."""
    return {
        "enabled": ENABLED,
        "ttl_seconds": TTL_SECONDS,
        "max_entries": MAX_ENTRIES,
        "file": CACHE_FILE or None,
    }


def make_key(request: Request, prompt_version: str) -> str:
    """Ключ кешу для запиту: однакові запити — однаковий ключ, будь-яка
    відмінність у тому, що впливає на відповідь, — інший."""
    provider = request.metadata.get("provider") or os.getenv("LLM_PROVIDER", "openai")
    material = {
        "provider": provider,
        "model": request.model or os.getenv("LLM_MODEL") or ("mock-1" if provider == "mock" else None),
        "prompt_version": prompt_version,
        "messages": request.messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    canonical = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not CACHE_FILE or not os.path.exists(CACHE_FILE):
        return
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        entries = data.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("cache entries must be an object")
        _entries.update(entries)
    except (OSError, ValueError, TypeError) as exc:
        _loaded = False
        raise RuntimeError(f"Unable to load response cache: {exc}") from exc


def _persist() -> None:
    if not CACHE_FILE:
        return
    path = os.path.abspath(CACHE_FILE)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, delete=False
        ) as file:
            temporary_path = file.name
            json.dump({"entries": _entries}, file, ensure_ascii=False, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _prune(now: float) -> bool:
    expired = [key for key, entry in _entries.items()
               if float(entry.get("expires_at", 0)) <= now]
    for key in expired:
        del _entries[key]
    return bool(expired)


def get(key: str) -> Result | None:
    """Результат за ключем, якщо він є й не прострочений; інакше `None`.
    Коли кеш вимкнено — завжди `None`."""
    global _hits, _misses
    if not ENABLED:
        return None
    with _lock:
        _load()
        changed = _prune(time.time())
        entry = _entries.get(key)
        if entry is None:
            _misses += 1
            if changed:
                _persist()
            return None
        try:
            result = Result(**entry["result"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid response cache entry") from exc
        _hits += 1
        if changed:
            _persist()
        return result


def contains(key: str) -> bool:
    """Check for a live entry without affecting operator-visible counters."""
    if not ENABLED:
        return False
    with _lock:
        _load()
        changed = _prune(time.time())
        present = key in _entries
        if changed:
            _persist()
        return present


def put(key: str, result: Result) -> None:
    """Зберегти результат. Коли кеш вимкнено — нічого не робити."""
    if not ENABLED or TTL_SECONDS <= 0 or MAX_ENTRIES <= 0:
        return
    with _lock:
        _load()
        now = time.time()
        _prune(now)
        _entries[key] = {
            "expires_at": now + TTL_SECONDS,
            "result": {
                "text": result.text,
                "provider": result.provider,
                "model": result.model,
                "usage": result.usage,
                "finish_reason": result.finish_reason,
                "elapsed": result.elapsed,
                "cost": result.cost,
                "cost_estimated": result.cost_estimated,
                "cached": False,
            },
        }
        while len(_entries) > MAX_ENTRIES:
            oldest = min(_entries, key=lambda item: _entries[item].get("expires_at", 0))
            del _entries[oldest]
        _persist()


def stats() -> dict:
    """Лічильники для `GET /api/usage` і сторінки: наприклад,
    `{"hits": 3, "misses": 7, "bypassed": 1, "entries": 6}`."""
    global _bypassed
    with _lock:
        _load()
        changed = _prune(time.time())
        if changed:
            _persist()
        return {
            "enabled": ENABLED,
            "hits": _hits,
            "misses": _misses,
            "bypassed": _bypassed,
            "entries": len(_entries),
        }


def bypassed() -> None:
    """Count a deliberate cache bypass requested by the operator."""
    global _bypassed
    with _lock:
        _bypassed += 1
