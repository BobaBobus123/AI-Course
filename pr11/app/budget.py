"""Денний ліміт витрат: скільки застосунок може витратити за добу.

Обмеження частоти з ПР10 рахує запити за хвилину і не завадить
витратити місячний бюджет рівномірним потоком. Тут рахуються гроші за
день на весь застосунок. Ліній дві: мʼяка — попередження на частці
`WARN_AT` (оператор бачить, журнал отримує `WARNING`), жорстка —
відмова після вичерпання.

Модуль знає лише суму, ліміт і день. Шлюз (`app/gateway.py`) викликає
`check()` перед платним звертанням до провайдера і `record(cost)`
після нього; веб-рівень показує `status()` на сторінці.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* де перевіряти: перед кожним платним викликом у шлюзі — обовʼязково
  (влучання в кеш безкоштовне й перевірки не потребує); чи ще й до
  запуску фонової роботи, щоб оператор дізнався одразу, а не подією
  через секунду;
* що з двома одночасними викликами на залишок бюджету: обидва пройдуть
  перевірку. Резервувати очікувану вартість наперед і уточнювати після
  — чи змиритися з невеликим перевищенням;
* стан на диску: лічильник у памʼяті обнулюється з перезапуском — і
  перезапуск «поповнює» бюджет. Файл у `state/` з датою і сумою,
  атомарний запис, як у ПР9;
* межа доби: за яким часом — локальним чи UTC; коли саме відновлюється
  ліміт і як це сказати операторові (`resets_at`);
* чернетка, яка вже генерується, коли бюджет вичерпався посеред неї:
  дати завершити чи перервати;
* вартість `None` (не визначено): рахувати як нуль, як оцінку, чи
  відмовляти, бо не знаємо, скільки витратили.
"""

import datetime
import json
import logging
import math
import os
import tempfile
import threading

from dotenv import load_dotenv

load_dotenv()

DAILY_USD = float(os.getenv("BUDGET_DAILY_USD", "1.00"))
WARN_AT = float(os.getenv("BUDGET_WARN_AT", "0.8"))
BUDGET_FILE = os.getenv("BUDGET_FILE", "state/budget.json")
_lock = threading.RLock()
_warned_day: str | None = None
_log = logging.getLogger("app.budget")


class BudgetExceeded(Exception):
    """Денний ліміт вичерпано. `spent`, `limit` — у доларах; `resets_at` —
    коли ліміт відновиться, у форматі ISO."""

    def __init__(self, spent: float, limit: float, resets_at: str) -> None:
        super().__init__(f"денний ліміт витрат {limit:.6f} $ вичерпано (витрачено {spent:.6f} $); "
                         f"відновиться {resets_at}")
        self.spent = spent
        self.limit = limit
        self.resets_at = resets_at


def settings() -> dict:
    """Чинні налаштування — для сторінки."""
    return {"daily_usd": DAILY_USD, "warn_at": WARN_AT, "file": BUDGET_FILE, "timezone": "UTC"}


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _read(day: str) -> float:
    if not BUDGET_FILE or not os.path.exists(BUDGET_FILE):
        return 0.0
    try:
        with open(BUDGET_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        if data.get("day") != day:
            return 0.0
        spent = float(data["spent"])
        if not math.isfinite(spent) or spent < 0:
            raise ValueError("invalid spent amount")
        return spent
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f"Unable to load daily budget state: {exc}") from exc


def _write(day: str, spent: float) -> None:
    if not BUDGET_FILE:
        return
    path = os.path.abspath(BUDGET_FILE)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, delete=False
        ) as file:
            temporary_path = file.name
            json.dump({"day": day, "spent": round(spent, 8)}, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _warn_if_needed(day: str, spent: float) -> bool:
    global _warned_day
    warning = DAILY_USD > 0 and spent >= DAILY_USD * WARN_AT
    if warning and _warned_day != day:
        _log.warning("budget.warning", extra={
            "day": day,
            "spent": round(spent, 8),
            "limit": DAILY_USD,
            "warn_at": WARN_AT,
        })
        _warned_day = day
    return warning


def _reset_time(now: datetime.datetime) -> str:
    tomorrow = (now + datetime.timedelta(days=1)).date()
    return datetime.datetime.combine(
        tomorrow, datetime.time.min, tzinfo=datetime.timezone.utc
    ).isoformat(timespec="seconds")


def check(expected_cost: float | None = None) -> None:
    """Переконатися, що платний виклик ще дозволений; інакше підняти
    `BudgetExceeded`. `expected_cost` — очікувана вартість, якщо ви
    вирішите резервувати її наперед."""
    now = _now()
    day = now.date().isoformat()
    expected = 0.0 if expected_cost is None else float(expected_cost)
    if not math.isfinite(expected) or expected < 0:
        raise ValueError("expected_cost must be a finite non-negative number")
    with _lock:
        spent = _read(day)
        _warn_if_needed(day, spent)
        if DAILY_USD <= 0 or spent + expected >= DAILY_USD:
            raise BudgetExceeded(spent, DAILY_USD, _reset_time(now))


def record(cost: float | None) -> None:
    """Зарахувати вартість виконаного звертання до витрат дня."""
    if cost is None:
        return
    amount = float(cost)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("cost must be a finite non-negative number")
    now = _now()
    day = now.date().isoformat()
    with _lock:
        spent = _read(day) + amount
        _write(day, spent)
        _warn_if_needed(day, spent)


def status() -> dict:
    """Стан для `GET /api/usage` і сторінки: наприклад,
    `{"day": "2026-10-23", "spent": 0.1234, "limit": 1.0, "remaining": 0.8766,
    "warning": False, "exceeded": False, "resets_at": "2026-10-24T00:00:00+03:00"}`."""
    now = _now()
    day = now.date().isoformat()
    with _lock:
        spent = _read(day)
        warning = _warn_if_needed(day, spent)
        exceeded = DAILY_USD <= 0 or spent >= DAILY_USD
        return {
            "day": day,
            "spent": round(spent, 8),
            "limit": DAILY_USD,
            "remaining": round(max(0.0, DAILY_USD - spent), 8),
            "warning": warning,
            "exceeded": exceeded,
            "resets_at": _reset_time(now),
        }
