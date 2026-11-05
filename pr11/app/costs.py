"""Estimate provider-call costs in USD from usage or text lengths.

Prices are configured per million tokens. Cached prompt tokens use their
separate rate when the provider reports them. If usage is unavailable, an
approximation is calculated from prompt and completion text and explicitly
marked as estimated. Mock calls are billable by default for local budget tests.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Ціни за мільйон токенів, у доларах.
PRICE_INPUT = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.30"))
PRICE_OUTPUT = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "2.50"))
_cached = os.getenv("LLM_PRICE_CACHED_INPUT_PER_1M", "")
PRICE_CACHED_INPUT = float(_cached) if _cached else PRICE_INPUT * 0.25
MOCK_BILLABLE = os.getenv("MOCK_BILLABLE", "1") == "1"


def settings() -> dict:
    """Чинні ціни — для сторінки й журналу."""
    return {
        "input_per_1m": PRICE_INPUT,
        "output_per_1m": PRICE_OUTPUT,
        "cached_input_per_1m": PRICE_CACHED_INPUT,
        "mock_billable": MOCK_BILLABLE,
        "currency": "USD",
    }


def estimate(usage: dict | None, *, provider: str | None = None, model: str | None = None,
             text: str | None = None, input_text: str | None = None) -> tuple[float | None, bool]:
    """Оцінити вартість одного звертання до провайдера.

    `usage` — те, що повернув провайдер (може бути `None`); `provider` і
    `model` — хто відповідав; `text` — текст відповіді, а `input_text` —
    текст повідомлень запиту, якщо доведеться оцінювати без `usage`.

    Повертає пару: вартість у доларах (або `None`, якщо її не визначити)
    і ознаку «це оцінка, а не розрахунок за фактичним `usage`».
    """
    if provider == "mock" and not MOCK_BILLABLE:
        return 0.0, False

    estimated = usage is None
    if usage is None:
        # A mixed-language approximation. Four characters per token is
        # conservative for Ukrainian prose while remaining reasonable for English.
        input_tokens = max(0, round(len(input_text or "") / 4))
        output_tokens = max(0, round(len(text or "") / 4))
    else:
        try:
            input_tokens = max(0, int(usage.get("prompt_tokens", 0) or 0))
            output_tokens = max(0, int(usage.get("completion_tokens", 0) or 0))
            cached_tokens = max(
                0,
                int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Invalid provider usage data") from exc
        cached_tokens = min(input_tokens, cached_tokens)

    if usage is None:
        cached_tokens = 0

    amount = (
        (input_tokens - cached_tokens) * PRICE_INPUT
        + cached_tokens * PRICE_CACHED_INPUT
        + output_tokens * PRICE_OUTPUT
    ) / 1_000_000
    return round(amount, 8), estimated
