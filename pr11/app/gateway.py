"""Заготовка з ПР10. Замініть цей файл своїм `pr10/app/gateway.py` і
допишіть те, що перелічено нижче.

Що додати в ПР11:

* журнал: запис на кожну спробу — початок і кінець — та підсумок на
  запит. Поля: ідентифікатори запиту й чернетки (з `Request.metadata`),
  клієнт, спроба, провайдер, модель, яка відповіла, версія prompt, час
  очікування в черзі одночасності, час до першого фрагмента, тривалість,
  статус або вид збою, запасний варіант, токени, вартість, кеш. Рівень:
  `INFO` для штатного, `WARNING` для повтору й запасного варіанту,
  `ERROR` для остаточного збою. Без тексту повідомлень і без ключа;
* кеш: перед першою спробою — `cache.get(cache.make_key(...))`.
  Влучання — результат із `cached=True` без звертання до провайдера,
  але фрагменти споживачеві все одно віддаються (одним шматком чи
  кількома — ваше рішення). Промах — після вдалої генерації
  `cache.put`. Прапорець `fresh` у `metadata` оминає читання кешу
  (а запис?). Запасний варіант і підставний провайдер — кешувати чи ні;
* бюджет: `budget.check()` перед звертанням до провайдера — не перед
  кешем. Вичерпано — `ProviderError(kind="budget", retryable=False)` з
  повідомленням для людини й часом відновлення, без повторів.
  `budget.record(cost)` після кожного платного звертання — включно зі
  спробами, що завершилися збоєм після того, як токени вже витрачено;
* вартість: `costs.estimate(usage, ...)` після кожного звертання → у
  `Result.cost`, `Result.cost_estimated` і в журнал; `usage` зі
  streaming — див. провайдера.

Шлюз і далі не знає, як влаштовані журнал, кеш, бюджет і ціни: він
викликає `cache.get/put`, `budget.check/record`, `costs.estimate` і
`logging` — і все.

Нижче — заготовка ПР10 без змін.

Шлюз до моделі: єдиний шлях, яким застосунок звертається до
провайдера.

Решта застосунку не викликає провайдерів напряму. Вона каже шлюзу
«згенеруй» — і отримує або результат, або контрольовану помилку. Між
цими двома подіями шлюз робить те, що не повинен робити ані провайдер,
ані веб-рівень, ані фонова задача:

* обирає провайдера за конфігурацією (`LLM_PROVIDER`) і запасний
  варіант (`LLM_FALLBACK`);
* тримає таймаути: на спробу, на бездіяльність у streaming, на весь
  запит разом зі спробами;
* повторює спроби — лише ті збої, які має сенс повторювати, з
  наростаючою паузою й випадковою часткою, не більше `MAX_ATTEMPTS`
  і не довше за `TOTAL_TIMEOUT`;
* обмежує кількість одночасних звертань до провайдера на весь
  застосунок (`MAX_CONCURRENT`);
* перериває роботу, коли той, хто просив, передумав;
* повідомляє, що відбувається: яка спроба, який провайдер, чи це
  запасний варіант — щоб сторінка могла це показати.

Шлюз не знає, що саме генерується (чернетка відповіді чи щось інше) і
звідки прийшов запит (сторінка, тест, скрипт). Він знає лише
`providers.base`.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* три таймаути — на спробу, на бездіяльність, на весь запит. Який
  ловить що, і що стається, коли спливає кожен із них посеред
  streaming, коли частину тексту вже віддано;
* які `kind` повторювати, а які ні; що робити з `retry_after` від
  провайдера, якщо він більший за ваш бюджет часу;
* чи повторювати спробу після того, як перший фрагмент уже віддано
  споживачеві. Якщо так — як сказати йому «починаємо спочатку»;
* коли вмикати запасний варіант: після всіх спроб, після першої, одразу
  на певних видах збою? Що з результату має сказати, що він запасний;
* що означає `MAX_CONCURRENT`, коли місця немає: чекати (скільки?) чи
  відмовляти одразу — і як це відрізняється від обмеження частоти для
  клієнтів у `app/limits.py`;
* як передавати фрагменти й стан назовні: зворотний виклик, генератор,
  черга? Як дізнаватися про скасування між фрагментами й між спробами;
* що потрапляє в підсумок для журналу: спроби, провайдер, модель, час
  до першого фрагмента, загальний час, токени.
"""

import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import replace
from typing import Callable

from dotenv import load_dotenv

from .providers.base import Provider, Request, Result
from . import budget, cache, costs
from .providers.base import Chunk, ProviderError

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai")
FALLBACK = os.getenv("LLM_FALLBACK", "none")

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
IDLE_TIMEOUT = float(os.getenv("LLM_IDLE_TIMEOUT", "8"))
TOTAL_TIMEOUT = float(os.getenv("LLM_TOTAL_TIMEOUT", "45"))
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "0.5"))
BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))
MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "2"))

# Параметри генерації — спільні для всіх провайдерів.
MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
_semaphore = threading.BoundedSemaphore(max(1, MAX_CONCURRENT))
_log = logging.getLogger("app.gateway")


def settings() -> dict:
    """Чинні налаштування шлюзу — для сторінки й журналу. Ключа тут
    немає і бути не може."""
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "backoff_base": BACKOFF_BASE,
        "backoff_max": BACKOFF_MAX,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str) -> Provider:
    """Зібрати провайдера за назвою: `openai` — з налаштувань доступу в
    `.env`, `mock` — підставний. Інша назва — та сама адреса й ключ, але
    інша модель (для запасного варіанту)."""
    if name == "mock":
        from .providers.mock import MockProvider
        return MockProvider()

    from .providers.openai_compat import OpenAICompatProvider

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )
    model = MODEL if name == "openai" else name
    return OpenAICompatProvider(base_url, api_key, model, TIMEOUT)


def preflight(sample: dict, fresh: bool = False) -> None:
    """Reject an already-exhausted paid request before accepting the job.

    Cache hits and explicitly free mock-only configurations remain available.
    The gateway repeats this check immediately before each provider call.
    """
    from .drafting import PROMPT_VERSION, build_request

    request = build_request(sample)
    request.metadata["provider"] = PROVIDER
    request.metadata["fresh"] = fresh
    if not fresh and cache.contains(cache.make_key(request, PROMPT_VERSION)):
        return
    provider_names = [PROVIDER]
    if FALLBACK and FALLBACK.lower() != "none" and FALLBACK != PROVIDER:
        provider_names.append(FALLBACK)
    if any(name != "mock" or costs.MOCK_BILLABLE for name in provider_names):
        budget.check()


def _cancelled_error() -> ProviderError:
    return ProviderError("cancelled", "Запит скасовано.", retryable=False)


def _next_chunk(iterator):
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


def _run_attempt(
    provider: Provider,
    request: Request,
    on_delta: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    deadline: float,
    metrics: dict | None = None,
) -> Result:
    started = time.monotonic()
    attempt_deadline = min(deadline, started + TIMEOUT)
    stream = provider.stream(request, is_cancelled)
    iterator = iter(stream)
    executor = ThreadPoolExecutor(max_workers=1)
    active_future = None
    text_parts: list[str] = []
    usage = None
    finish_reason = None
    try:
        while True:
            if is_cancelled():
                raise _cancelled_error()
            remaining = min(deadline, attempt_deadline) - time.monotonic()
            if remaining <= 0:
                raise ProviderError(
                    "timeout",
                    "Час очікування відповіді вичерпано.",
                    retryable=time.monotonic() < deadline,
                )
            future = executor.submit(_next_chunk, iterator)
            active_future = future
            try:
                done, chunk = future.result(timeout=min(IDLE_TIMEOUT, remaining))
            except TimeoutError as exc:
                future.cancel()
                raise ProviderError(
                    "timeout",
                    "Час очікування фрагмента відповіді вичерпано.",
                    retryable=time.monotonic() < deadline,
                ) from exc
            if done:
                break
            if chunk.text:
                if metrics is not None and metrics.get("first_token_ms") is None:
                    metrics["first_token_ms"] = round((time.monotonic() - started) * 1000, 1)
                text_parts.append(chunk.text)
                on_delta(chunk.text)
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
        if is_cancelled():
            raise _cancelled_error()
        return Result(
            text="".join(text_parts),
            provider=provider.name,
            model=getattr(provider, "model", request.model or MODEL or ""),
            usage=usage,
            finish_reason=finish_reason or "stop",
            elapsed=time.monotonic() - started,
        )
    finally:
        close = getattr(stream, "close", None)
        if close:
            if active_future is not None and not active_future.done():
                active_future.add_done_callback(lambda _: close())
            else:
                close()
        executor.shutdown(wait=False, cancel_futures=True)


def _prompt_text(request: Request) -> str:
    return "\n".join(str(message.get("content", "")) for message in request.messages)


def _token_fields(usage: dict | None) -> dict:
    if not usage:
        return {"input": None, "output": None}
    return {
        "input": usage.get("prompt_tokens"),
        "output": usage.get("completion_tokens"),
    }


def _record_attempt(
    *,
    request: Request,
    draft_id: str | None,
    attempt: int,
    provider: str,
    model: str,
    started: float,
    usage: dict | None,
    result_text: str,
    error: ProviderError | None,
    fallback: bool,
    metrics: dict,
) -> tuple[float | None, bool]:
    cost, estimated = costs.estimate(
        usage,
        provider=provider,
        model=model,
        text=result_text,
        input_text=_prompt_text(request),
    )
    budget.record(cost)
    fields = {
        "request_id": request.metadata.get("request_id"),
        "sample_id": request.metadata.get("sample_id"),
        "draft_id": draft_id,
        "attempt": attempt,
        "provider": provider,
        "model": model,
        "prompt_version": request.metadata.get("prompt_version", "unknown"),
        "duration_ms": round((time.monotonic() - started) * 1000, 1),
        "first_token_ms": metrics.get("first_token_ms"),
        "queue_ms": metrics.get("queue_ms"),
        "status": "failed" if error else "success",
        "kind": error.kind if error else "none",
        "tokens": _token_fields(usage),
        "cost": cost,
        "cost_estimated": estimated,
        "cached": False,
        "fallback": fallback,
    }
    if error and error.retryable:
        _log.warning("llm.attempt", extra=fields)
    elif error:
        _log.error("llm.attempt", extra=fields)
    else:
        _log.info("llm.attempt", extra=fields)
    return cost, estimated


def generate(request: Request, on_delta: Callable[[str], None],
             on_status: Callable[[dict], None], is_cancelled: Callable[[], bool]) -> Result:
    """Згенерувати відповідь, дотримуючись таймаутів, спроб, обмеження
    одночасності й запасного варіанту.

    `on_delta` отримує кожен фрагмент тексту, `on_status` — зміни стану
    (наприклад, `{"attempt": 2, "provider": "openai"}` або
    `{"fallback": True}`), `is_cancelled` опитується між фрагментами й
    між спробами. Сигнатура — відправна точка: якщо вам зручніше
    генератор чи черга, змініть і її, і `app/jobs.py`.

    Повертає `Result` або підіймає `ProviderError` — уже остаточний,
    після всіх спроб і запасного варіанту. Інших винятків назовні не
    виходить.
    """
    started = time.monotonic()
    deadline = started + TOTAL_TIMEOUT
    if is_cancelled():
        raise _cancelled_error()

    request.metadata.setdefault("provider", PROVIDER)
    prompt_version = str(request.metadata.get("prompt_version", "unknown"))
    key = cache.make_key(request, prompt_version)
    if request.metadata.get("fresh"):
        cache.bypassed()
        _log.info("cache.bypassed", extra={
            "request_id": request.metadata.get("request_id"),
            "draft_id": request.metadata.get("draft_id"),
            "sample_id": request.metadata.get("sample_id"),
            "cached": False,
            "prompt_version": prompt_version,
        })
    else:
        cached_result = cache.get(key)
        if cached_result is not None:
            _log.info("cache.hit", extra={
                "request_id": request.metadata.get("request_id"),
                "draft_id": request.metadata.get("draft_id"),
                "sample_id": request.metadata.get("sample_id"),
                "provider": cached_result.provider,
                "model": cached_result.model,
                "prompt_version": prompt_version,
                "cached": True,
            })
            return replace(cached_result, cost=0.0, cost_estimated=False, cached=True)
        _log.info("cache.miss", extra={
            "request_id": request.metadata.get("request_id"),
            "draft_id": request.metadata.get("draft_id"),
            "sample_id": request.metadata.get("sample_id"),
            "provider": request.metadata.get("provider"),
            "model": request.model or MODEL,
            "prompt_version": prompt_version,
            "cached": False,
        })

    queue_started = time.monotonic()
    if not _semaphore.acquire(timeout=max(0.0, min(TIMEOUT, TOTAL_TIMEOUT))):
        raise ProviderError(
            "unavailable",
            "Зараз усі місця для запитів зайняті. Спробуйте пізніше.",
            retryable=True,
        )

    try:
        queue_ms = round((time.monotonic() - queue_started) * 1000, 1)
        if is_cancelled():
            raise _cancelled_error()
        provider_names = [PROVIDER]
        if FALLBACK and FALLBACK.lower() != "none" and FALLBACK != PROVIDER:
            provider_names.append(FALLBACK)
        last_error: ProviderError | None = None
        attempt_number = 0
        total_cost = 0.0
        any_cost_estimated = False

        for provider_index, provider_name in enumerate(provider_names):
            fallback = provider_index > 0
            provider = make_provider(provider_name)
            attempts = 1 if fallback else max(1, MAX_ATTEMPTS)
            if fallback:
                _log.warning("llm.fallback", extra={
                    "request_id": request.metadata.get("request_id"),
                    "draft_id": request.metadata.get("draft_id"),
                    "sample_id": request.metadata.get("sample_id"),
                    "provider": provider_name,
                    "prompt_version": prompt_version,
                    "cached": False,
                })
                on_status({"fallback": True, "provider": provider_name})

            for attempt in range(1, attempts + 1):
                if is_cancelled():
                    raise _cancelled_error()
                if time.monotonic() >= deadline:
                    last_error = ProviderError(
                        "timeout", "Перевищено загальний час виконання запиту.", retryable=False
                    )
                    break
                attempt_number += 1
                on_status({
                    "attempt": attempt_number,
                    "provider": provider_name,
                    "fallback": fallback,
                })
                if provider_name != "mock" or costs.MOCK_BILLABLE:
                    try:
                        budget.check()
                    except budget.BudgetExceeded as exc:
                        raise ProviderError(
                            "budget",
                            str(exc),
                            retryable=False,
                        ) from exc

                attempt_started = time.monotonic()
                parts: list[str] = []
                metrics = {"first_token_ms": None, "queue_ms": queue_ms}
                attempt_request = replace(
                    request,
                    model=getattr(provider, "model", request.model) if fallback else request.model,
                    metadata={**request.metadata, "provider": provider_name},
                )

                def collect_delta(delta: str) -> None:
                    parts.append(delta)
                    on_delta(delta)

                result = None
                error = None
                try:
                    result = _run_attempt(
                        provider, attempt_request, collect_delta, is_cancelled, deadline, metrics
                    )
                    if result.finish_reason == "length":
                        raise ProviderError(
                            "incomplete",
                            "Провайдер завершив відповідь до її закінчення.",
                            retryable=False,
                        )
                except ProviderError as exc:
                    error = exc
                except Exception as exc:
                    error = ProviderError(
                        "unknown",
                        "Під час генерації сталася внутрішня помилка.",
                        retryable=False,
                        detail=type(exc).__name__,
                    )

                usage = result.usage if result else None
                model = result.model if result else getattr(provider, "model", request.model or MODEL or "")
                text = result.text if result else "".join(parts)
                cost, estimated = _record_attempt(
                    request=request,
                    draft_id=request.metadata.get("draft_id"),
                    attempt=attempt_number,
                    provider=provider_name,
                    model=model,
                    started=attempt_started,
                    usage=usage,
                    result_text=text,
                    error=error,
                    fallback=fallback,
                    metrics=metrics,
                )
                if cost is not None:
                    total_cost += cost
                any_cost_estimated = any_cost_estimated or estimated
                on_status({
                    "attempt_cost": cost,
                    "cost_estimated": estimated,
                    "usage": usage,
                    "cached": False,
                })

                if error is None and result is not None:
                    result.cost = round(total_cost, 8)
                    result.cost_estimated = any_cost_estimated
                    result.cached = False
                    result.usage = result.usage or {}
                    cache.put(key, result)
                    on_status({
                        "done": True,
                        "provider": result.provider,
                        "model": result.model,
                        "attempt": attempt_number,
                        "fallback": fallback,
                        "usage": result.usage,
                        "cost": result.cost,
                        "cost_estimated": result.cost_estimated,
                        "cached": False,
                    })
                    return result

                last_error = error or ProviderError("unknown", "Невідома помилка.", retryable=False)
                if last_error.kind == "cancelled":
                    raise last_error
                if parts:
                    on_status({"reset": True})
                if not last_error.retryable or attempt >= attempts:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                delay = min(
                    BACKOFF_MAX,
                    BACKOFF_BASE * (2 ** (attempt - 1)),
                )
                if last_error.retry_after is not None:
                    delay = max(delay, last_error.retry_after)
                delay = min(delay + random.uniform(0, 0.25), remaining)
                _log.warning("llm.retry", extra={
                    "request_id": request.metadata.get("request_id"),
                    "draft_id": request.metadata.get("draft_id"),
                    "sample_id": request.metadata.get("sample_id"),
                    "attempt": attempt_number,
                    "provider": provider_name,
                    "kind": last_error.kind,
                    "delay_ms": round(delay * 1000, 1),
                    "prompt_version": prompt_version,
                    "cached": False,
                })
                on_status({
                    "retry": True,
                    "attempt": attempt_number,
                    "delay": delay,
                    "kind": last_error.kind,
                })
                end = time.monotonic() + delay
                while time.monotonic() < end:
                    if is_cancelled():
                        raise _cancelled_error()
                    time.sleep(min(0.05, end - time.monotonic()))

            if is_cancelled():
                raise _cancelled_error()

        if last_error is not None:
            raise last_error
        raise ProviderError(
            "unknown", "Не вдалося отримати відповідь від провайдера.", retryable=False
        )
    finally:
        _semaphore.release()
