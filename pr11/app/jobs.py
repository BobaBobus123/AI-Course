"""Заготовка з ПР10. Замініть цей файл своїм `pr10/app/jobs.py` і
допишіть те, що перелічено нижче.

Що додати в ПР11:

* ідентифікатор запиту: фонова робота виконується в іншому потоці, і
  контекстна змінна `telemetry.request_id_var` там порожня. Передайте
  ідентифікатор явно (у `Draft`, у `Request.metadata`, в `extra`
  кожного запису) або запустіть роботу в копії контексту
  (`contextvars.copy_context().run`). Усі записи чернетки мають містити
  `draft_id` і `request_id`;
* журнал: `draft.created` (звернення, клієнт, fresh) і `draft.done` /
  `draft.failed` / `draft.cancelled` (спроб, провайдер, кеш, вартість,
  тривалість). Текст чернетки — ні;
* прапорець `fresh` з `POST /api/drafts`: `start(sample, client,
  fresh=False)` → `Request.metadata` → шлюз оминає кеш. Сигнатуру
  заготовки нижче вже змінено — змініть і свою;
* у `Draft`, знімку й події `done` — `cached`, `cost`, `cost_estimated`:
  сторінка показує їх, якщо вони є;
* відмова через бюджет: шлюз підіймає `ProviderError(kind="budget")` —
  це подія `error` з `retryable=False` і часом відновлення в
  повідомленні. Чи перевіряти бюджет ще й до запуску роботи, щоб
  оператор дізнався одразу, — див. `main.py`.

Нижче — заготовка ПР10 без змін.

Фонові задачі: чернетка генерується не в HTTP-запиті, а окремо.

`POST /api/drafts` повертається одразу з номером чернетки, а робота
триває у фоні: сторінка дізнається про стан через потік подій
(`GET /api/drafts/{id}/events`) або знімок (`GET /api/drafts/{id}`) і
може скасувати її (`POST /api/drafts/{id}/cancel`). Так чернетка не
прив'язана до одного HTTP-зʼєднання: сторінку можна перезавантажити,
а потім відкрити ту саму чернетку й побачити все, що встигло статися.

Модуль не знає, як влаштований шлюз і що саме генерується; він знає
лише `Draft`, події і те, що робота виконується не в потоці запиту.

Що тут дано готовим — реєстр чернеток і події. `emit` додає подію до
журналу чернетки й будить тих, хто чекає; `subscribe` віддає події
від указаного номера і далі в міру надходження, доки чернетка не
дійде до кінцевого статусу. Веб-рівень перетворює це на
server-sent events; реалізація фонової роботи до цього не торкається.

Заготовки — `start` і `cancel`. Рішення з розділу 2 практичної роботи:

* у чому виконувати: потік на чернетку, пул потоків, asyncio-задача?
  Що обмежує кількість одночасних робіт — це чи шлюз;
* порядок дій у фоновій роботі: яку подію й який статус виставити
  спершу, що робити з фрагментами, що — з результатом і з помилкою.
  Жоден виняток не має лишити чернетку в стані `running`;
* скасування — спільне: фонова робота сама перевіряє прапорець між
  фрагментами й спробами. Хто його ставить, де він живе, що
  відбувається з текстом, який уже надіслано, і чи може скасована
  чернетка потім стати `done`, якщо потік завершився пізніше;
* `cancel` для чернетки, яка вже завершена чи не існує, — помилка якого
  роду і з яким HTTP-статусом;
* що буває зі старими чернетками: реєстр у памʼяті росте з кожною.
  Скільки тримати і коли прибирати.

Події, які розуміє сторінка (інші вона показує як є):

    status     {"status": "running", "attempt": 1, "provider": "openai",
                "fallback": false, "detail": "…"}
    delta      {"text": "…"}                      приріст тексту
    done       {"text": "…", "provider": "…", "model": "…",
                "usage": {...}, "elapsed": {...}, "attempts": 2,
                "fallback": false, "warnings": ["…"]}
    error      {"kind": "timeout", "message": "…", "retryable": true}
    cancelled  {}

Поля — відправна точка; змінюючи їх, змініть і сторінку.
"""

import contextvars
import datetime
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from . import telemetry
from .drafting import build_request, check
from .gateway import generate
from .providers.base import ProviderError

# Статуси, після яких подій більше не буде.
TERMINAL = {"done", "failed", "cancelled"}
_logger = logging.getLogger("app.jobs")
DRAFTS_FILE = os.getenv("DRAFTS_FILE", "state/drafts.json")
MAX_SAVED_DRAFTS = max(1, int(os.getenv("DRAFTS_MAX_ENTRIES", "100")))


@dataclass
class Draft:
    """Стан однієї чернетки — те, що зберігається в реєстрі і йде на
    сторінку як знімок.

    `status` — `queued`, `running`, `done`, `failed`, `cancelled`.
    `text` — накопичений текст. `attempts`, `provider`, `model`,
    `fallback` — звідки взявся результат. `error` — остаточна помилка,
    якщо є: `kind`, `message`, `retryable`. `elapsed` — секунди, наприклад
    `{"first_token": 0.8, "total": 4.1}`. `warnings` — зауваження до
    чернетки з `drafting.check`. `events` — журнал подій для потоку;
    у знімок на сторінку не входить.
    """

    draft_id: str
    request_id: str
    client: str
    trace_id: str = ""
    status: str = "queued"
    text: str = ""
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    error: dict | None = None
    usage: dict | None = None
    elapsed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cost: float | None = None
    cost_estimated: bool = False
    cached: bool = False
    created_at: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_lock = threading.Lock()
_changed = threading.Condition(_lock)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def settings() -> dict:
    return {"file": DRAFTS_FILE or None, "max_entries": MAX_SAVED_DRAFTS}


def _saved_draft(draft: Draft) -> dict:
    return {
        "draft_id": draft.draft_id,
        "request_id": draft.request_id,
        "trace_id": draft.trace_id,
        "status": draft.status,
        "text": draft.text,
        "attempts": draft.attempts,
        "provider": draft.provider,
        "model": draft.model,
        "fallback": draft.fallback,
        "error": draft.error,
        "usage": draft.usage,
        "elapsed": draft.elapsed,
        "warnings": draft.warnings,
        "cost": draft.cost,
        "cost_estimated": draft.cost_estimated,
        "cached": draft.cached,
        "created_at": draft.created_at,
        "updated_at": draft.updated_at,
        "events": draft.events,
    }


def _persist_locked() -> None:
    if not DRAFTS_FILE:
        return
    while len(_order) > MAX_SAVED_DRAFTS:
        oldest_terminal = next(
            (draft_id for draft_id in reversed(_order)
             if _drafts[draft_id].status in TERMINAL),
            None,
        )
        if oldest_terminal is None:
            break
        _order.remove(oldest_terminal)
        del _drafts[oldest_terminal]

    path = os.path.abspath(DRAFTS_FILE)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, delete=False
        ) as file:
            temporary_path = file.name
            json.dump(
                {"drafts": [_saved_draft(_drafts[draft_id]) for draft_id in _order]},
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _load_saved() -> None:
    if not DRAFTS_FILE or not os.path.exists(DRAFTS_FILE):
        return
    try:
        with open(DRAFTS_FILE, "r", encoding="utf-8") as file:
            records = json.load(file)["drafts"]
        if not isinstance(records, list):
            raise ValueError("drafts must be a list")
        for saved in records:
            if not isinstance(saved, dict):
                raise ValueError("saved draft must be an object")
            draft = Draft(
                draft_id=str(saved["draft_id"]),
                request_id=str(saved["request_id"]),
                client="restored",
                trace_id=str(saved.get("trace_id", "")),
                status=str(saved.get("status", "failed")),
                text=str(saved.get("text", "")),
                attempts=int(saved.get("attempts", 0)),
                provider=saved.get("provider"),
                model=saved.get("model"),
                fallback=bool(saved.get("fallback", False)),
                error=saved.get("error"),
                usage=saved.get("usage"),
                elapsed=saved.get("elapsed") or {},
                warnings=saved.get("warnings") or [],
                cost=saved.get("cost"),
                cost_estimated=bool(saved.get("cost_estimated", False)),
                cached=bool(saved.get("cached", False)),
                created_at=str(saved.get("created_at", "")),
                updated_at=str(saved.get("updated_at", "")),
                events=saved.get("events") or [],
            )
            if draft.status not in TERMINAL:
                draft.status = "failed"
                draft.error = {
                    "kind": "server_restarted",
                    "message": "Сервер перезапустився до завершення генерації.",
                    "retryable": True,
                    "request_id": draft.trace_id,
                    "cost": draft.cost,
                    "cost_estimated": draft.cost_estimated,
                }
                draft.updated_at = _now()
                draft.events.append({
                    "id": len(draft.events) + 1,
                    "event": "error",
                    "data": draft.error,
                })
            _drafts[draft.draft_id] = draft
            _order.append(draft.draft_id)
        _order.sort(key=lambda draft_id: _drafts[draft_id].created_at, reverse=True)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f"Unable to load saved drafts from {DRAFTS_FILE}: {exc}") from exc


def new_draft(request_id: str, client: str, trace_id: str | None = None) -> Draft:
    """Створити чернетку зі статусом `queued` і зареєструвати її.
    Роботу не запускає — це справа `start`."""
    draft = Draft(draft_id=uuid.uuid4().hex[:8], request_id=request_id, client=client,
                  trace_id=trace_id or telemetry.request_id_var.get() or telemetry.new_request_id(),
                  created_at=_now(), updated_at=_now())
    with _lock:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)
        _persist_locked()
    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def list_drafts(limit: int = 50) -> list[Draft]:
    """Чернетки, нові першими."""
    with _lock:
        return [_drafts[i] for i in _order[:limit]]


def emit(draft: Draft, event: str, data: dict | None = None) -> None:
    """Додати подію до журналу чернетки й розбудити підписників.

    Статус і текст чернетки ця функція не змінює — це робить фонова
    робота разом із подією, під тим самим замком, якщо їй потрібна
    узгодженість між знімком і потоком.
    """
    with _changed:
        draft.events.append({"id": len(draft.events) + 1, "event": event, "data": data or {}})
        draft.updated_at = _now()
        _persist_locked()
        _changed.notify_all()


def subscribe(draft_id: str, since: int = 0, heartbeat: float = 15.0) -> Iterator[dict | None]:
    """Віддавати події чернетки, починаючи з номера `since + 1`.

    Спершу — ті, що вже є, потім — нові в міру надходження. Завершується,
    коли чернетка в кінцевому статусі й усі події віддано, або коли
    чернетки немає. Якщо подій немає довше за `heartbeat` секунд, віддає
    `None`: веб-рівень надсилає порожній коментар, щоб зʼєднання не
    закрив проксі і щоб помітити, що клієнт відключився.
    """
    while True:
        with _changed:
            draft = _drafts.get(draft_id)
            if draft is None:
                return
            pending = draft.events[since:]
            if not pending:
                if draft.status in TERMINAL:
                    return
                _changed.wait(timeout=heartbeat)
                pending = draft.events[since:]
        if not pending:
            yield None
            continue
        for event in pending:
            yield event
        since += len(pending)


def start(sample: dict, client: str, fresh: bool = False) -> Draft:
    """Створити чернетку для звернення `sample` і запустити генерацію у
    фоні. Повертає чернетку одразу — найімовірніше, ще `queued`.
    `fresh` — оминути кеш (кнопка «Повторити»)."""
    trace_id = telemetry.request_id_var.get() or telemetry.new_request_id()
    draft = new_draft(str(sample.get("id", "")), client, trace_id)
    _logger.info("draft.created", extra={
        "request_id": trace_id,
        "draft_id": draft.draft_id,
        "fresh": bool(fresh),
    })
    context = contextvars.copy_context()
    thread = threading.Thread(
        target=context.run,
        args=(_run_draft, draft, sample, bool(fresh)),
        name=f"draft-{draft.draft_id}",
        daemon=True,
    )
    try:
        thread.start()
    except RuntimeError as exc:
        with _changed:
            draft.status = "failed"
            draft.error = {
                "kind": "internal_error",
                "message": "Не вдалося запустити фонову генерацію.",
                "retryable": False,
                "request_id": trace_id,
            }
            draft.events.append({
                "id": len(draft.events) + 1,
                "event": "error",
                "data": draft.error,
            })
            _persist_locked()
            _changed.notify_all()
        _logger.error("draft.failed", extra={
            "request_id": trace_id,
            "draft_id": draft.draft_id,
            "kind": "internal_error",
        })
        raise RuntimeError("Unable to start background draft generation") from exc
    return draft


def cancel(draft_id: str) -> Draft:
    """Попросити фонову роботу зупинитися. Повертає чернетку; якщо її
    немає або вона вже завершена — вирішуєте ви."""
    with _changed:
        draft = _drafts.get(draft_id)
        if draft is None:
            raise KeyError("Чернетку не знайдено")
        if draft.status in TERMINAL:
            raise ValueError("Чернетка вже завершена")
        draft.cancel_event.set()
        draft.status = "cancelled"
        draft.updated_at = _now()
        draft.events.append({
            "id": len(draft.events) + 1,
            "event": "cancelled",
            "data": {},
        })
        _persist_locked()
        _changed.notify_all()
    _logger.info("draft.cancelled", extra={
        "request_id": draft.trace_id,
        "draft_id": draft.draft_id,
    })
    return draft


def _run_draft(draft: Draft, sample: dict, fresh: bool) -> None:
    started = time.perf_counter()
    request_token = telemetry.request_id_var.set(draft.trace_id)
    draft_token = telemetry.draft_id_var.set(draft.draft_id)
    try:
        with _changed:
            if draft.cancel_event.is_set():
                return
            draft.status = "running"
            draft.events.append({
                "id": len(draft.events) + 1,
                "event": "status",
                "data": {"status": "running"},
            })
            _persist_locked()
            _changed.notify_all()

        request = build_request(sample)
        request.metadata.update({
            "request_id": draft.trace_id,
            "draft_id": draft.draft_id,
            "sample_id": draft.request_id,
            "fresh": fresh,
        })

        def on_delta(text: str) -> None:
            if not text:
                return
            with _changed:
                if draft.status != "running":
                    return
                draft.text += text
                draft.events.append({
                    "id": len(draft.events) + 1,
                    "event": "delta",
                    "data": {"text": text},
                })
                _changed.notify_all()

        def on_status(data: dict) -> None:
            with _changed:
                if data.get("attempt_cost") is not None:
                    draft.cost = round((draft.cost or 0.0) + float(data["attempt_cost"]), 8)
                    draft.cost_estimated = draft.cost_estimated or bool(data.get("cost_estimated"))
                    if data.get("usage") is not None:
                        draft.usage = data["usage"]
                if draft.status != "running":
                    return
                if "attempt" in data and not data.get("retry") and not data.get("done"):
                    draft.attempts = max(draft.attempts, int(data["attempt"]))
                if "provider" in data:
                    draft.provider = data["provider"]
                if "fallback" in data:
                    draft.fallback = bool(data["fallback"])
                if data.get("reset"):
                    draft.text = ""
                draft.events.append({
                    "id": len(draft.events) + 1,
                    "event": "status",
                    "data": data,
                })
                _persist_locked()
                _changed.notify_all()

        result = generate(request, on_delta, on_status, draft.cancel_event.is_set)
        warnings = check(result.text, sample)
        with _changed:
            if draft.status != "running":
                return
            draft.text = result.text
            draft.provider = result.provider
            draft.model = result.model
            draft.usage = result.usage
            draft.cost = result.cost
            draft.cost_estimated = result.cost_estimated
            draft.cached = result.cached
            draft.warnings = warnings
            draft.elapsed = {"total": round(time.perf_counter() - started, 2)}
            draft.status = "done"
            draft.events.append({
                "id": len(draft.events) + 1,
                "event": "done",
                "data": {
                    "text": draft.text,
                    "provider": draft.provider,
                    "model": draft.model,
                    "usage": draft.usage or {},
                    "elapsed": draft.elapsed,
                    "attempts": draft.attempts,
                    "fallback": draft.fallback,
                    "warnings": draft.warnings,
                    "cached": draft.cached,
                    "cost": draft.cost,
                    "cost_estimated": draft.cost_estimated,
                },
            })
            _persist_locked()
            _changed.notify_all()
        _logger.info("draft.done", extra={
            "request_id": draft.trace_id,
            "draft_id": draft.draft_id,
            "attempts": draft.attempts,
            "provider": draft.provider,
            "model": draft.model,
            "cached": draft.cached,
            "cost": draft.cost,
            "cost_estimated": draft.cost_estimated,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        })
    except ProviderError as exc:
        with _changed:
            if draft.status == "cancelled":
                return
            draft.status = "failed"
            draft.error = {
                "kind": exc.kind,
                "message": exc.message,
                "retryable": exc.retryable,
                "request_id": draft.trace_id,
                "cost": draft.cost,
                "cost_estimated": draft.cost_estimated,
            }
            draft.events.append({
                "id": len(draft.events) + 1,
                "event": "error",
                "data": draft.error,
            })
            _persist_locked()
            _changed.notify_all()
        _logger.warning("draft.failed", extra={
            "request_id": draft.trace_id,
            "draft_id": draft.draft_id,
            "kind": exc.kind,
            "attempts": draft.attempts,
            "cost": draft.cost,
            "cost_estimated": draft.cost_estimated,
        })
    except Exception as exc:
        with _changed:
            if draft.status == "cancelled":
                return
            draft.status = "failed"
            draft.error = {
                "kind": "internal_error",
                "message": "Не вдалося згенерувати чернетку.",
                "retryable": False,
                "request_id": draft.trace_id,
                "cost": draft.cost,
                "cost_estimated": draft.cost_estimated,
            }
            draft.events.append({
                "id": len(draft.events) + 1,
                "event": "error",
                "data": draft.error,
            })
            _persist_locked()
            _changed.notify_all()
        _logger.error("draft.failed", extra={
            "request_id": draft.trace_id,
            "draft_id": draft.draft_id,
            "kind": "internal_error",
            "error_type": type(exc).__name__,
            "cost": draft.cost,
            "cost_estimated": draft.cost_estimated,
        })
    finally:
        telemetry.draft_id_var.reset(draft_token)
        telemetry.request_id_var.reset(request_token)


with _changed:
    _load_saved()
    if _drafts:
        _persist_locked()
