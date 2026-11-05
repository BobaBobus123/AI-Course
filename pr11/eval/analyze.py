"""Аналіз журналу: повільні, невдалі й повторні виклики, витрати, кеш.

Запуск із папки pr11:

    python eval/analyze.py                       # logs/app.jsonl, усі розділи
    python eval/analyze.py logs/app.jsonl --since 2026-10-23 --section slow

Читання файлу, відбір за датою і допоміжні функції готові. Розділи
звіту — заготовки: поля, за якими вони працюють, — ваші, бо назви
подій і полів ви обрали, коли писали записи в шлюзі й фонових задачах.
`summary` працює на будь-якому журналі й показує, що в ньому взагалі є.

Орієнтир для кожного розділу — питання, на яке він відповідає; воно в
докстрінгу. Відповідь — числа й кілька рядків журналу, а не переказ.
"""

import argparse
import datetime
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path, since: str | None = None) -> list[dict]:
    """Прочитати JSON lines. Рядки, які не розбираються, рахуються й
    пропускаються; `since` — дата або час у форматі ISO, від якого брати
    записи (порівнюється з полем `ts` як рядок)."""
    records: list[dict] = []
    broken = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                broken += 1
                continue
            if since and str(record.get("ts", "")) < since:
                continue
            records.append(record)
    if broken:
        print(f"(пропущено рядків, що не розбираються: {broken})", file=sys.stderr)
    return records


def percentile(values: list[float], p: float) -> float | None:
    """p-й процентиль (0 ≤ p ≤ 1) за найближчим рангом; `None` для
    порожнього списку."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def group(records: list[dict], key: str) -> dict:
    """Згрупувати записи за значенням поля."""
    groups: dict = defaultdict(list)
    for record in records:
        groups[record.get(key)].append(record)
    return groups


def summary(records: list[dict]) -> None:
    """Що взагалі є в журналі: скільки записів, за який час, яких подій і
    рівнів скільки. Працює на будь-якому журналі."""
    if not records:
        print("записів немає")
        return
    stamps = [r.get("ts", "") for r in records]
    print(f"записів: {len(records)}; від {min(stamps)} до {max(stamps)}")
    print("рівні:", dict(Counter(r.get("level") for r in records)))
    print("події:")
    for event, count in Counter(r.get("event") for r in records).most_common():
        print(f"  {count:>6}  {event}")


def _attempts(records: list[dict]) -> list[dict]:
    return [record for record in records if record.get("event") == "llm.attempt"]


def _number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _money(value: float | None) -> str:
    return "н/д" if value is None else f"${value:.8f}"


def _timestamp(record: dict) -> datetime.datetime | None:
    try:
        value = datetime.datetime.fromisoformat(str(record.get("ts", "")).replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def _cost_total(records: list[dict]) -> float:
    return sum(_number(record.get("cost")) or 0.0 for record in records)


def slow(records: list[dict]) -> None:
    """Які виклики найповільніші й що таке «повільно» в цих числах?

    Розподіл тривалості звертань до провайдера (медіана, p95, максимум),
    а не середнє; десятка найдовших із ідентифікаторами, провайдером,
    моделлю, спробою; де пройшов час — у черзі, до першого фрагмента чи
    в генерації; розріз за провайдером і годиною.
    """
    attempts = _attempts(records)
    measured = [(record, _number(record.get("duration_ms"))) for record in attempts]
    measured = [(record, duration) for record, duration in measured if duration is not None]
    if not measured:
        print("звертань до провайдера з duration_ms немає")
        return
    durations = [duration for _, duration in measured]
    print(
        f"спроб: {len(durations)}; медіана {statistics.median(durations):.1f} ms; "
        f"p95 {percentile(durations, .95):.1f} ms; максимум {max(durations):.1f} ms"
    )
    print("найповільніші спроби:")
    for record, duration in sorted(measured, key=lambda pair: pair[1], reverse=True)[:10]:
        print(
            f"  {duration:>9.1f} ms request={record.get('request_id', '-')} "
            f"draft={record.get('draft_id', '-')} attempt={record.get('attempt', '-')} "
            f"provider={record.get('provider', '-')} model={record.get('model', '-')}"
        )
    print("медіана за провайдером:")
    providers = defaultdict(list)
    hours = defaultdict(list)
    for record, duration in measured:
        providers[record.get("provider", "unknown")].append(duration)
        stamp = _timestamp(record)
        if stamp:
            hours[stamp.strftime("%Y-%m-%d %H:00")].append(duration)
    for provider, values in sorted(providers.items(), key=lambda item: str(item[0])):
        print(f"  {provider}: {statistics.median(values):.1f} ms ({len(values)} спроб)")
    if hours:
        print("медіана за годиною:")
        for hour, values in sorted(hours.items()):
            print(f"  {hour}: {statistics.median(values):.1f} ms ({len(values)} спроб)")
    queue_values = [_number(record.get("queue_ms")) for record, _ in measured]
    queue_values = [value for value in queue_values if value is not None]
    first_values = [_number(record.get("first_token_ms")) for record, _ in measured]
    first_values = [value for value in first_values if value is not None]
    if queue_values:
        print(f"очікування слота: медіана {statistics.median(queue_values):.1f} ms")
    if first_values:
        print(f"до першого токена: медіана {statistics.median(first_values):.1f} ms")


def failed(records: list[dict]) -> None:
    """Що й чому не вдалося?

    Невдалі спроби за видом збою; остаточні збої чернеток; частка
    чернеток із невдалою першою спробою, які все ж завершилися успіхом
    (чи окупилися повтори); вартість спроб, які нічого не дали.
    """
    attempts = _attempts(records)
    failed_attempts = [record for record in attempts if record.get("status") == "failed"]
    if not attempts:
        print("спроб провайдера немає")
        return
    kinds = Counter(record.get("kind", "unknown") for record in failed_attempts)
    print(f"невдалих спроб: {len(failed_attempts)} з {len(attempts)}")
    print("за видом збою:", dict(kinds) if kinds else "немає")
    terminal = [record for record in records if record.get("event") == "draft.failed"]
    print(f"остаточно невдалих чернеток: {len(terminal)}")
    failed_by_draft = Counter(record.get("draft_id") for record in failed_attempts)
    done = [record for record in records if record.get("event") == "draft.done"]
    recovered = [
        record for record in done
        if failed_by_draft.get(record.get("draft_id"), 0) > 0
    ]
    failed_first = sum(
        1 for draft_id, count in failed_by_draft.items()
        if count and any(
            record.get("draft_id") == draft_id and int(record.get("attempt") or 0) > 1
            for record in attempts
        )
    )
    print(f"чернеток, урятованих після невдалої спроби: {len(recovered)}")
    print(f"чернеток із першою невдалою спробою: {failed_first}")
    costs = [record for record in failed_attempts if _number(record.get("cost")) is not None]
    print(f"вартість невдалих спроб: {_money(_cost_total(costs))}; без cost: {len(failed_attempts) - len(costs)}")


def repeated(records: list[dict]) -> None:
    """Що ми робили двічі?

    Той самий ключ кешу кілька разів за короткий час (кандидати на кеш
    і питання «чому»); чернетки з кількома спробами (повтори шлюзу);
    клієнти з серіями замовлень за хвилину.
    """
    attempts = _attempts(records)
    by_draft = defaultdict(list)
    by_sample = defaultdict(list)
    for record in attempts:
        if record.get("draft_id"):
            by_draft[record["draft_id"]].append(record)
        if record.get("sample_id"):
            by_sample[record["sample_id"]].append(record)
    retries = [(draft_id, items) for draft_id, items in by_draft.items() if len(items) > 1]
    print(f"чернеток із повторними спробами шлюзу: {len(retries)}")
    for draft_id, items in sorted(retries, key=lambda item: len(item[1]), reverse=True)[:10]:
        print(f"  draft={draft_id}: {len(items)} спроб, kinds={[item.get('kind') for item in items]}")
    repeated_samples = [
        (sample_id, items) for sample_id, items in by_sample.items() if len(items) > 1
    ]
    print(f"зразків, надісланих провайдеру більше одного разу: {len(repeated_samples)}")
    for sample_id, items in sorted(repeated_samples, key=lambda item: len(item[1]), reverse=True)[:10]:
        print(f"  sample={sample_id}: {len(items)} спроб")
    events = [
        record for record in records
        if record.get("event") in ("draft.created", "cache.hit", "cache.miss", "cache.bypassed")
        and record.get("sample_id")
    ]
    by_sample_event = defaultdict(list)
    for record in events:
        stamp = _timestamp(record)
        if stamp:
            by_sample_event[record["sample_id"]].append(stamp.timestamp())
    bursts = []
    for sample_id, stamps in by_sample_event.items():
        stamps.sort()
        max_count = 0
        left = 0
        for right, stamp in enumerate(stamps):
            while stamp - stamps[left] > 60:
                left += 1
            max_count = max(max_count, right - left + 1)
        if max_count > 1:
            bursts.append((sample_id, max_count))
    print("повтори одного sample_id за хвилину (не клієнти; ідентифікатори клієнтів не журналюються):")
    if not bursts:
        print("  немає")
    for sample_id, count in sorted(bursts, key=lambda item: item[1], reverse=True)[:10]:
        print(f"  sample={sample_id}: до {count} подій/хв")


def spend(records: list[dict]) -> None:
    """Скільки це коштувало?

    Вартість і токени за день, за модель, за клієнта; середня вартість
    чернетки; частка вартості, яка припала на спроби зі збоєм і на
    запасний варіант; скільки записів без вартості або з оцінкою.
    """
    attempts = _attempts(records)
    if not attempts:
        print("звертань до провайдера немає")
        return
    total = _cost_total(attempts)
    missing = sum(record.get("cost") is None for record in attempts)
    estimated = sum(bool(record.get("cost_estimated")) for record in attempts)
    tokens_in = tokens_out = 0
    by_model = defaultdict(float)
    by_day = defaultdict(float)
    failed_cost = fallback_cost = 0.0
    per_draft = defaultdict(float)
    for record in attempts:
        cost = _number(record.get("cost")) or 0.0
        by_model[record.get("model", "unknown")] += cost
        stamp = _timestamp(record)
        if stamp:
            by_day[stamp.date().isoformat()] += cost
        token_data = record.get("tokens") or {}
        tokens_in += int(token_data.get("input") or 0)
        tokens_out += int(token_data.get("output") or 0)
        if record.get("status") == "failed":
            failed_cost += cost
        if record.get("fallback"):
            fallback_cost += cost
        if record.get("draft_id"):
            per_draft[record["draft_id"]] += cost
    print(
        f"provider attempts: {len(attempts)}; витрати: {_money(total)}; "
        f"вхідні токени: {tokens_in}; вихідні: {tokens_out}"
    )
    print(f"середня вартість чернетки за спробами: {_money(total / len(per_draft)) if per_draft else 'н/д'}")
    print(f"невдалі спроби: {_money(failed_cost)}; fallback: {_money(fallback_cost)}")
    print(f"без вартості: {missing}; оцінених: {estimated}")
    print("за моделлю:")
    for model, amount in sorted(by_model.items(), key=lambda item: str(item[0])):
        print(f"  {model}: {_money(amount)}")
    print("за днем:")
    for day, amount in sorted(by_day.items()):
        print(f"  {day}: {_money(amount)}")


def cache_report(records: list[dict]) -> None:
    """Що дав кеш?

    Влучання, промахи, обходи (`fresh`) і частка влучань; скільки
    звертань до провайдера й грошей заощаджено; які звернення
    повторювалися найчастіше.
    """
    hits = [record for record in records if record.get("event") == "cache.hit"]
    misses = [record for record in records if record.get("event") == "cache.miss"]
    bypassed = [record for record in records if record.get("event") == "cache.bypassed"]
    attempts = _attempts(records)
    denominator = len(hits) + len(misses)
    ratio = len(hits) / denominator * 100 if denominator else 0.0
    by_sample_cost = defaultdict(list)
    for record in attempts:
        if record.get("sample_id") and record.get("status") == "success":
            cost = _number(record.get("cost"))
            if cost is not None:
                by_sample_cost[record["sample_id"]].append(cost)
    average_costs = [cost for costs in by_sample_cost.values() for cost in costs]
    default_cost = statistics.mean(average_costs) if average_costs else 0.0
    saved = 0.0
    unknown_hits = 0
    hit_samples = Counter()
    for record in hits:
        sample_id = record.get("sample_id")
        if sample_id:
            hit_samples[sample_id] += 1
            values = by_sample_cost.get(sample_id)
        else:
            values = None
        if values:
            saved += statistics.mean(values)
        elif default_cost:
            saved += default_cost
        else:
            unknown_hits += 1
    print(
        f"влучання: {len(hits)}; промахи: {len(misses)}; обходи fresh: {len(bypassed)}; "
        f"hit ratio: {ratio:.1f}%"
    )
    print(f"викликів провайдера: {len(attempts)}; оцінка збережених витрат кешем: {_money(saved)}")
    if unknown_hits:
        print(f"влучань без бази для оцінки заощаджень: {unknown_hits}")
    if hit_samples:
        print("найчастіше кешовані зразки:")
        for sample_id, count in hit_samples.most_common(10):
            print(f"  {sample_id}: {count}")


SECTIONS = {
    "summary": summary,
    "slow": slow,
    "failed": failed,
    "repeated": repeated,
    "spend": spend,
    "cache": cache_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", default="logs/app.jsonl")
    parser.add_argument("--since", help="брати записи від цієї дати/часу (ISO), наприклад 2026-10-23")
    parser.add_argument("--section", choices=["all", *SECTIONS], default="all")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"журналу немає: {path}", file=sys.stderr)
        return 1
    records = load(path, args.since)

    names = list(SECTIONS) if args.section == "all" else [args.section]
    for name in names:
        print(f"\n=== {name}")
        try:
            SECTIONS[name](records)
        except NotImplementedError as exc:
            print(f"(заготовка: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
