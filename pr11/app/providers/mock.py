"""Заготовка з ПР10. Замініть цей файл своїм `pr10/app/providers/mock.py`.

Що перевірити в ПР11: підставний провайдер має повертати правдоподібний
`usage` (наприклад, за кількістю слів у запиті й відповіді), інакше
вартість, бюджет і порівняння з кешем не перевірити без ключа. За
замовчуванням він рахується за цінами справжнього (`MOCK_BILLABLE=1`).

Нижче — заготовка ПР10 без змін.

Підставний провайдер: поводиться як справжній, але без мережі й
ключа — і збоїть тоді, коли ви просите.

Навіщо він потрібен:

* застосунок запускається й демонструється без ключа і без доступу до
  провайдера — у поїзді, на захисті, в автоматичних тестах;
* збої відтворюються на вимогу. Дочекатися справжнього 429 чи обриву
  посеред streaming можна довго; тут вони стаються за налаштуванням;
* безкоштовний ліміт ключа не витрачається на перевірку кнопок.

Поводитися він має так само, як справжній: ті самі `Result`, `Chunk`,
`ProviderError`; фрагменти — із затримкою, а не всі одразу, інакше
сторінка не покаже, як виглядає streaming. Текст відповіді не мусить
бути розумним — досить правдоподібного: у `samples/requests.json` для
кожного звернення є `reply_example`, а яке саме звернення зараз
обробляється, видно з `Request.metadata`.

Клас нижче — заготовка. Рішення з розділу 2 практичної роботи:

* звідки текст: `reply_example` за `metadata["request_id"]`, щось
  складене з останнього повідомлення користувача чи сталий рядок — і
  що робити, коли звернення невідоме;
* як відтворити кожен вид збою з `MOCK_FAILURE_KIND`: `timeout` —
  мовчати довше за таймаут чи одразу підняти `ProviderError("timeout")`?
  Перше чесніше, бо перевіряє ваш таймаут, а не лише обробку помилки;
  `rate_limit` — з яким `retry_after`; `mid_stream` — обрив після
  кількох фрагментів, коли частину тексту вже показано;
* випадковість: `MOCK_FAILURE_RATE=0.3` дає різні прогони — зручно для
  демонстрації, незручно для тестів. Чи потрібен спосіб зробити збій
  детермінованим (наприклад, через `metadata`);
* чи має підставний провайдер поважати скасування: якщо споживач
  перестав читати фрагменти, затримки мають припинитися.

Значення нижче читаються з `.env`; їхній зміст описано в
`.env.example`.
"""

import os
import random
import time
from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")


class MockProvider:
    """Провайдер без моделі: канонічні тексти, затримки, збої на вимогу."""

    name = "mock"
    model = "mock-1"

    def complete(self, request: Request) -> Result:
        """Повернути відповідь цілком — після тих самих затримок і з
        тими самими збоями, що й у `stream`."""
        started = time.monotonic()
        self._wait(FIRST_DELAY_MS / 1000, lambda: False)
        if random.random() < FAILURE_RATE and FAILURE_KIND != "mid_stream":
            self._raise_failure()
        text = self._get_text(request)
        return Result(
            text=text,
            provider=self.name,
            model=self.model,
            usage=self._usage(request, text),
            finish_reason="stop",
            elapsed=time.monotonic() - started,
        )

    def stream(self, request: Request, is_cancelled=None) -> Iterator[Chunk]:
        """Віддавати текст фрагментами з затримкою `DELAY_MS` між ними."""
        return self._stream(request, is_cancelled or (lambda: False))

    def _get_text(self, request: Request) -> str:
        from ..samples import get

        sample_id = request.metadata.get("sample_id") or request.metadata.get("request_id")
        sample = get(str(sample_id)) if sample_id else None
        if sample:
            return str(sample.get("reply_example") or "Дякуємо за звернення. Ми уточнимо інформацію.")
        return "Дякуємо за звернення. Ми уточнимо інформацію."

    @staticmethod
    def _usage(request: Request, text: str) -> dict:
        input_count = sum(len(str(item.get("content", "")).split()) for item in request.messages)
        output_count = len(text.split())
        return {
            "prompt_tokens": max(1, input_count),
            "completion_tokens": max(1, output_count),
            "total_tokens": max(2, input_count + output_count),
        }

    @staticmethod
    def _wait(seconds: float, is_cancelled) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if is_cancelled():
                raise ProviderError("cancelled", "Запит скасовано.")
            time.sleep(min(0.05, end - time.monotonic()))
        if is_cancelled():
            raise ProviderError("cancelled", "Запит скасовано.")

    @staticmethod
    def _raise_failure() -> None:
        kind = FAILURE_KIND
        retryable = kind in ("unavailable", "timeout", "rate_limit")
        raise ProviderError(
            kind,
            "Підставний провайдер імітував помилку.",
            retryable=retryable,
            retry_after=1 if kind == "rate_limit" else None,
        )

    def _stream(self, request: Request, is_cancelled) -> Iterator[Chunk]:
        self._wait(FIRST_DELAY_MS / 1000, is_cancelled)
        fail = random.random() < FAILURE_RATE
        if fail and FAILURE_KIND != "mid_stream":
            self._raise_failure()
        text = self._get_text(request)
        words = text.split()
        mid_stream = fail and FAILURE_KIND == "mid_stream"
        for index, word in enumerate(words):
            if index:
                self._wait(DELAY_MS / 1000, is_cancelled)
            if mid_stream and index >= max(1, len(words) // 2):
                raise ProviderError(
                    "unavailable",
                    "З'єднання перервано під час відповіді.",
                    retryable=True,
                )
            yield Chunk(
                text=word + (" " if index < len(words) - 1 else ""),
                usage=self._usage(request, text) if index == len(words) - 1 and not mid_stream else None,
                finish_reason="stop" if index == len(words) - 1 and not mid_stream else None,
            )
        if not words:
            yield Chunk(text="", usage=self._usage(request, text), finish_reason="stop")
