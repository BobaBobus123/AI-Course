"""Заготовка з ПР10. Замініть цей файл своїм
`pr10/app/providers/openai_compat.py`.

Що перевірити в ПР11: `usage`. Для вартості потрібні токени на вході й
на виході, а в режимі streaming SDK віддає їх лише в останньому
фрагменті й лише з `stream_options={"include_usage": True}` — і не
кожен провайдер це підтримує (скрипт перевірки середовища покаже).
Якщо `usage` в потоці немає, `Chunk.usage` лишається `None`, і вартість
оцінює шлюз через `costs.estimate` за текстом. Кешовані провайдером
вхідні токени — у `usage.prompt_tokens_details.cached_tokens`, якщо є;
передайте їх далі як є. Журнал у провайдері не потрібен: його веде
шлюз.

Нижче — заготовка ПР10 без змін.

Справжній провайдер: OpenAI-сумісний API — той самий, що в ПР3–ПР9.

Єдиний модуль застосунку, який імпортує SDK `openai`. Його справа —
перетворити наш `Request` на виклик API, а відповідь і збої API — на
`Result`, `Chunk` і `ProviderError` з `base.py`. Що робити зі збоєм —
повторити, зачекати, перейти на запасний варіант — тут не вирішується:
це справа шлюзу.

Клас нижче — заготовка. Рішення з розділу 2 практичної роботи:

* який виняток SDK у який `kind` перетворюється і який із них
  `retryable`: `APITimeoutError`, `RateLimitError` (429),
  `APIConnectionError`, `InternalServerError` (5xx),
  `AuthenticationError` (401), `BadRequestError` (400). Де взяти
  `Retry-After`, якщо провайдер його надіслав;
* таймаут: SDK приймає його при створенні клієнта і на окремий виклик
  (`with_options`). Що передавати — `LLM_TIMEOUT` на спробу чи щось
  інше — і хто рахує таймаут бездіяльності в streaming: SDK такого не
  вміє, це робиться у споживача фрагментів;
* `max_retries` SDK за замовчуванням не нульовий — SDK сам мовчки
  повторює деякі запити. Залишити це йому чи вимкнути й повторювати у
  шлюзі, де видно спроби, бюджет і запасний варіант;
* streaming: `stream=True` віддає обʼєкт, по якому йдуть фрагментами
  (`choices[0].delta.content`). Як закрити його, якщо споживач
  перестав читати (скасування, таймаут бездіяльності), і чи приходить
  `usage` в останньому фрагменті (`stream_options`), а якщо ні —
  що записати;
* що з відповіді потрапляє в `detail` для журналу, а що — у `message`
  для людини. Ключ і повний текст запиту — ні туди, ні туди.

Обробку збоїв із ПР3–ПР9 перенесіть сюди — але тепер вона не підіймає
власний `LLMError`, а зводить усе до `ProviderError`.
"""

import time
from typing import Iterator

from .base import Chunk, ProviderError, Request, Result
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)


class OpenAICompatProvider:
    """Провайдер поверх OpenAI-сумісного API."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )

    @staticmethod
    def _provider_error(error: Exception) -> ProviderError:
        if isinstance(error, (APITimeoutError,)):
            kind, retryable = "timeout", True
        elif isinstance(error, RateLimitError):
            kind, retryable = "rate_limit", True
        elif isinstance(error, (APIConnectionError, InternalServerError)):
            kind, retryable = "unavailable", True
        elif isinstance(error, AuthenticationError):
            kind, retryable = "auth", False
        elif isinstance(error, BadRequestError):
            kind, retryable = "bad_request", False
        elif isinstance(error, APIStatusError):
            status = error.status_code
            kind = (
                "rate_limit" if status == 429 else
                "timeout" if status == 408 else
                "unavailable" if status >= 500 else
                "auth" if status in (401, 403) else
                "bad_request" if status >= 400 else
                "unknown"
            )
            retryable = status in (408, 429) or status >= 500
        else:
            kind, retryable = "unknown", False
        retry_after = None
        response = getattr(error, "response", None)
        if response is not None:
            try:
                retry_after = max(0.0, float(response.headers.get("retry-after")))
            except (TypeError, ValueError):
                pass
        detail = type(error).__name__
        status = getattr(error, "status_code", None)
        if status is not None:
            detail += f", status={status}"
        return ProviderError(
            kind,
            "Під час звернення до провайдера сталася помилка.",
            retryable=retryable,
            retry_after=retry_after,
            detail=detail,
        )

    def complete(self, request: Request) -> Result:
        """Отримати відповідь цілком."""
        started = time.monotonic()
        model = request.model or self.model
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                timeout=self.timeout,
            )
            choice = response.choices[0]
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return Result(
                text=choice.message.content or "",
                provider=self.name,
                model=response.model or model,
                usage=usage,
                finish_reason=choice.finish_reason,
                elapsed=time.monotonic() - started,
            )
        except Exception as error:
            raise self._provider_error(error) from error

    def stream(self, request: Request, is_cancelled=None) -> Iterator[Chunk]:
        """Віддавати фрагменти відповіді в міру надходження."""
        is_cancelled = is_cancelled or (lambda: False)
        model = request.model or self.model
        stream = None
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                timeout=self.timeout,
                stream=True,
                stream_options={"include_usage": True},
            )
            for part in stream:
                if is_cancelled():
                    raise ProviderError("cancelled", "Запит скасовано.")
                usage = None
                if part.usage:
                    usage = {
                        "prompt_tokens": part.usage.prompt_tokens,
                        "completion_tokens": part.usage.completion_tokens,
                        "total_tokens": part.usage.total_tokens,
                    }
                text = ""
                finish_reason = None
                if part.choices:
                    choice = part.choices[0]
                    text = choice.delta.content or ""
                    finish_reason = choice.finish_reason
                yield Chunk(text=text, usage=usage, finish_reason=finish_reason)
        except ProviderError:
            raise
        except Exception as error:
            raise self._provider_error(error) from error
        finally:
            if stream is not None:
                stream.close()
