from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from .images import prepare_image
from .llm import extract_invoice_data
from .rules import validate_invoice


@dataclass
class Result:
    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict[str, Any] | None = None
    issues: list[dict[str, str]] = field(default_factory=list)
    image: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    elapsed: dict[str, float] = field(default_factory=dict)
    usage: dict[str, Any] | None = None


def decide(
    document: dict[str, Any], issues: list[dict[str, str]]
) -> tuple[str, list[str]]:
    doc_type = str(document.get("doc_type") or "").casefold()
    if "рахунок" not in doc_type:
        return "reject", [
            "Документ не розпізнано як рахунок на оплату "
            "(можливо, це видаткова накладна)."
        ]

    reasons = [issue["message"] for issue in issues if issue.get("message")]
    for field in ("number", "date", "total_no_vat", "vat", "total_to_pay"):
        if not document.get(field):
            reasons.append(f"Не заповнено обов'язкове поле «{field}».")

    for party_name, fields in (
        ("supplier", ("name", "code", "iban")),
        ("buyer", ("name", "code")),
    ):
        party = document.get(party_name)
        if not isinstance(party, dict):
            reasons.append(f"Не розпізнано дані: {party_name}.")
            continue
        for field in fields:
            if not party.get(field):
                reasons.append(f"Не заповнено обов'язкове поле «{party_name}.{field}».")

    items = document.get("items")
    if not isinstance(items, list) or not items:
        reasons.append("У рахунку не розпізнано жодної позиції.")
    else:
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                reasons.append(f"Не розпізнано позицію {index + 1}.")
                continue
            for field in ("name", "quantity", "price", "sum"):
                if not item.get(field):
                    reasons.append(
                        f"Не заповнено поле «items.{index}.{field}»."
                    )

    if reasons:
        return "review", list(dict.fromkeys(reasons))
    return "auto", []


def process(content: bytes) -> Result:
    """Prepare, extract, validate, and decide without hiding service failures."""
    elapsed: dict[str, float] = {}

    started = perf_counter()
    data_url, image_meta = prepare_image(content)
    elapsed["prepare"] = round(perf_counter() - started, 3)

    started = perf_counter()
    extracted_data, llm_meta = extract_invoice_data(data_url)
    elapsed["extraction"] = round(perf_counter() - started, 3)

    started = perf_counter()
    issues = validate_invoice(extracted_data)
    elapsed["checks"] = round(perf_counter() - started, 3)

    document = extracted_data.model_dump()
    decision, reasons = decide(document, issues)
    return Result(
        decision=decision,
        reasons=reasons,
        document=document,
        issues=issues,
        image=image_meta,
        model=llm_meta.get("model"),
        elapsed=elapsed,
        usage=llm_meta,
    )
