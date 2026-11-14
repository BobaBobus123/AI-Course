import json
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from pathlib import Path
from typing import Any

from app.schema import InvoiceSchema


REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
CENT = Decimal("0.01")


class ReferenceDataError(RuntimeError):
    """Required company or supplier reference data could not be loaded."""


def _read_reference(filename: str) -> Any:
    path = REFERENCE_DIR / filename
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceDataError(
            f"Не вдалося завантажити довідник {filename}."
        ) from exc


def load_suppliers() -> list[dict[str, Any]]:
    suppliers = _read_reference("suppliers.json")
    if not isinstance(suppliers, list) or not all(
        isinstance(supplier, dict)
        and supplier.get("name")
        and supplier.get("code")
        and supplier.get("iban")
        and isinstance(supplier.get("vat_payer"), bool)
        for supplier in suppliers
    ):
        raise ReferenceDataError("Довідник постачальників має бути списком.")
    return suppliers


def load_company() -> dict[str, Any]:
    company = _read_reference("company.json")
    if (
        not isinstance(company, dict)
        or not company.get("name")
        or not company.get("code")
    ):
        raise ReferenceDataError(
            "Довідник компанії має містити назву та код."
        )
    return company


def _add_issue(
    issues: list[dict[str, str]],
    field: str,
    rule: str,
    message: str,
    severity: str = "error",
) -> None:
    issues.append(
        {"field": field, "rule": rule, "message": message, "severity": severity}
    )


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_code(value: Any) -> str:
    return re.sub(r"\s+", "", _text(value))


def _normalize_iban(value: Any) -> str:
    return re.sub(r"\s+", "", _text(value)).upper()


def _normalize_name(value: Any) -> str:
    value = unicodedata.normalize("NFKC", _text(value)).casefold()
    value = value.translate(
        str.maketrans(
            {
                "a": "а",
                "b": "в",
                "c": "с",
                "e": "е",
                "h": "н",
                "i": "і",
                "k": "к",
                "m": "м",
                "o": "о",
                "p": "р",
                "t": "т",
                "x": "х",
            }
        )
    )
    value = value.replace("ʼ", "'").replace("’", "'").replace("`", "'")
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)


def is_valid_iban(value: Any) -> bool:
    iban = _normalize_iban(value)
    if not re.fullmatch(r"UA[0-9]{27}", iban):
        return False

    rearranged = iban[4:] + iban[:4]
    remainder = 0
    for character in rearranged:
        expanded = str(ord(character) - ord("A") + 10) if character.isalpha() else character
        for digit in expanded:
            remainder = (remainder * 10 + int(digit)) % 97
    return remainder == 1


def is_valid_edrpou(value: Any) -> bool:
    code = _normalize_code(value)
    if not re.fullmatch(r"[0-9]{8}", code):
        return False

    digits = [int(digit) for digit in code]
    weights = list(range(1, 8)) if digits[0] <= 2 else [7, 1, 2, 3, 4, 5, 6]
    checksum = sum(digit * weight for digit, weight in zip(digits[:7], weights)) % 11
    if checksum == 10:
        shifted_weights = (
            [3, 4, 5, 6, 7, 8, 9]
            if digits[0] <= 2
            else [9, 3, 4, 5, 6, 7, 8]
        )
        checksum = (
            sum(
                digit * weight
                for digit, weight in zip(digits[:7], shifted_weights)
            )
            % 11
        )
    return checksum < 10 and checksum == digits[7]


def is_valid_rnokpp(value: Any) -> bool:
    code = _normalize_code(value)
    if not re.fullmatch(r"[0-9]{10}", code):
        return False

    digits = [int(digit) for digit in code]
    weights = [-1, 5, 7, 9, 4, 6, 10, 5, 7]
    checksum = sum(
        digit * weight for digit, weight in zip(digits[:9], weights)
    ) % 11 % 10
    return checksum == digits[9]


def is_valid_tax_code(value: Any) -> bool:
    code = _normalize_code(value)
    if len(code) == 8:
        return is_valid_edrpou(code)
    if len(code) == 10:
        return is_valid_rnokpp(code)
    return False


def _decimal(value: Any) -> Decimal:
    normalized = re.sub(r"[\s\u00a0]", "", _text(value)).replace(",", ".")
    if not re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", normalized):
        raise InvalidOperation
    if sum(character.isdigit() for character in normalized) > 32:
        raise InvalidOperation
    result = Decimal(normalized)
    if not result.is_finite():
        raise InvalidOperation
    return result


def _money(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 80
        return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _parse_date(value: Any) -> date | None:
    text = _text(value)
    for date_format in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            pass
    return None


def validate_invoice(invoice: InvoiceSchema) -> list[dict[str, str]]:
    """Return structured errors; any error prevents automatic approval."""
    issues: list[dict[str, str]] = []
    company = load_company()
    suppliers = load_suppliers()

    required_fields = {
        "number": (invoice.number, "Номер рахунку"),
        "date": (invoice.date, "Дата рахунку"),
        "total_no_vat": (invoice.total_no_vat, "Сума без ПДВ"),
        "vat": (invoice.vat, "Сума ПДВ"),
        "total_to_pay": (invoice.total_to_pay, "Сума до сплати"),
    }
    for field, (value, label) in required_fields.items():
        if not _text(value):
            _add_issue(
                issues,
                field,
                "required",
                f"Не заповнено обов'язкове поле «{label}».",
            )

    if invoice.date and not _parse_date(invoice.date):
        _add_issue(issues, "date", "date_format", "Некоректна дата рахунку.")
    if invoice.due_date and not _parse_date(invoice.due_date):
        _add_issue(
            issues, "due_date", "date_format", "Некоректна дата строку оплати."
        )
    invoice_date = _parse_date(invoice.date)
    due_date = _parse_date(invoice.due_date)
    if invoice_date and due_date and due_date < invoice_date:
        _add_issue(
            issues,
            "due_date",
            "date_order",
            "Строк оплати не може передувати даті рахунку.",
        )

    if not invoice.items:
        _add_issue(
            issues, "items", "required", "У рахунку не розпізнано жодної позиції."
        )

    for party_name, party in (("supplier", invoice.supplier), ("buyer", invoice.buyer)):
        if not _text(party.name):
            _add_issue(
                issues,
                f"{party_name}.name",
                "required",
                "Не розпізнано назву контрагента.",
            )
        if not _text(party.code):
            _add_issue(
                issues,
                f"{party_name}.code",
                "required",
                "Не розпізнано код ЄДРПОУ/РНОКПП.",
            )
        elif not is_valid_tax_code(party.code):
            _add_issue(
                issues,
                f"{party_name}.code",
                "tax_code_checksum",
                "Контрольна цифра ЄДРПОУ/РНОКПП не пройшла перевірку.",
            )

    if not _text(invoice.supplier.iban):
        _add_issue(
            issues,
            "supplier.iban",
            "required",
            "Не розпізнано IBAN постачальника.",
        )
    elif not is_valid_iban(invoice.supplier.iban):
        _add_issue(
            issues,
            "supplier.iban",
            "iban_checksum",
            "IBAN постачальника має некоректний формат або контрольну суму.",
        )

    company_code = _normalize_code(company.get("code"))
    buyer_code = _normalize_code(invoice.buyer.code)
    if buyer_code and buyer_code != company_code:
        _add_issue(
            issues,
            "buyer.code",
            "buyer_reference",
            "Покупець у рахунку не збігається з компанією «Сузір'я Рітейл».",
        )
    expected_buyer_name = _normalize_name(company.get("name"))
    if (
        _text(invoice.buyer.name)
        and _normalize_name(invoice.buyer.name) != expected_buyer_name
    ):
        _add_issue(
            issues,
            "buyer.name",
            "buyer_reference",
            "Назва покупця не збігається з company.json.",
        )

    supplier_code = _normalize_code(invoice.supplier.code)
    supplier_reference = next(
        (
            supplier
            for supplier in suppliers
            if _normalize_code(supplier.get("code")) == supplier_code
        ),
        None,
    )
    if supplier_code and supplier_reference is None:
        _add_issue(
            issues,
            "supplier.code",
            "supplier_reference",
            "Постачальника немає у довіднику перевірених контрагентів.",
        )
    elif supplier_reference is not None:
        if _normalize_name(invoice.supplier.name) != _normalize_name(
            supplier_reference.get("name")
        ):
            _add_issue(
                issues,
                "supplier.name",
                "supplier_reference",
                "Назва постачальника не збігається з довідником.",
            )
        expected_iban = _normalize_iban(supplier_reference.get("iban"))
        actual_iban = _normalize_iban(invoice.supplier.iban)
        if actual_iban and actual_iban != expected_iban:
            _add_issue(
                issues,
                "supplier.iban",
                "supplier_reference",
                "IBAN постачальника не збігається з довідником.",
            )

    item_amounts: list[Decimal] = []
    for index, item in enumerate(invoice.items):
        item_prefix = f"items.{index}"
        for field in ("name", "quantity", "price", "sum"):
            if not _text(getattr(item, field)):
                _add_issue(
                    issues,
                    f"{item_prefix}.{field}",
                    "required",
                    f"Не розпізнано поле «{field}» позиції {index + 1}.",
                )
        if not all(
            _text(getattr(item, field)) for field in ("quantity", "price", "sum")
        ):
            continue
        try:
            quantity = _decimal(item.quantity)
            price = _decimal(item.price)
            amount = _decimal(item.sum)
            if quantity <= 0 or price < 0 or amount < 0:
                raise InvalidOperation
        except (InvalidOperation, TypeError):
            _add_issue(
                issues,
                item_prefix,
                "number_format",
                f"Некоректні числові дані у позиції {index + 1}.",
            )
            continue

        item_amounts.append(_money(amount))
        with localcontext() as context:
            context.prec = 80
            calculated_amount = _money(quantity * price)
        if calculated_amount != _money(amount):
            _add_issue(
                issues,
                f"{item_prefix}.sum",
                "line_math",
                f"Сума позиції {index + 1} не дорівнює кількості, помноженій на ціну "
                f"({calculated_amount:.2f} очікується).",
            )

    values: dict[str, Decimal] = {}
    for field, (raw_value, label) in (
        ("total_no_vat", (invoice.total_no_vat, "суми без ПДВ")),
        ("vat", (invoice.vat, "ПДВ")),
        ("total_to_pay", (invoice.total_to_pay, "суми до сплати")),
    ):
        if not _text(raw_value):
            continue
        try:
            values[field] = _money(_decimal(raw_value))
            if values[field] < 0:
                raise InvalidOperation
        except (InvalidOperation, TypeError):
            _add_issue(
                issues,
                field,
                "number_format",
                f"Некоректний числовий формат {label}.",
            )

    if len(item_amounts) == len(invoice.items) and item_amounts and "total_no_vat" in values:
        with localcontext() as context:
            context.prec = 80
            calculated_total = _money(sum(item_amounts, Decimal("0")))
        if calculated_total != values["total_no_vat"]:
            _add_issue(
                issues,
                "total_no_vat",
                "subtotal_math",
                f"Сума позицій {calculated_total:.2f} не збігається із сумою без ПДВ "
                f"{values['total_no_vat']:.2f}.",
            )

    if supplier_reference is not None and "total_no_vat" in values and "vat" in values:
        vat_rate = Decimal("0.20") if supplier_reference.get("vat_payer") else Decimal("0")
        with localcontext() as context:
            context.prec = 80
            expected_vat = _money(values["total_no_vat"] * vat_rate)
        if values["vat"] != expected_vat:
            _add_issue(
                issues,
                "vat",
                "vat_rate",
                f"ПДВ має дорівнювати {expected_vat:.2f} за ставкою "
                f"{vat_rate * 100:.0f}%.",
            )

    if "total_no_vat" in values and "vat" in values and "total_to_pay" in values:
        with localcontext() as context:
            context.prec = 80
            expected_total = _money(values["total_no_vat"] + values["vat"])
        if values["total_to_pay"] != expected_total:
            _add_issue(
                issues,
                "total_to_pay",
                "total_math",
                f"Сума до сплати має дорівнювати {expected_total:.2f}.",
            )

    return issues
