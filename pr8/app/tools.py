"""Allow-listed, schema-validated customer tools for the shop service."""

import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from shop import service


load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
try:
    SHOP_DELAY_MS = int(os.getenv("SHOP_DELAY_MS", "0") or 0)
    SHOP_FAILURE_RATE = float(os.getenv("SHOP_FAILURE_RATE", "0") or 0)
    SHOP_TIMEOUT_SECONDS = float(os.getenv("SHOP_TIMEOUT_SECONDS", "10"))
except ValueError as exc:
    raise ValueError("Invalid shop delay, failure rate, or timeout configuration.") from exc
if (
    SHOP_DELAY_MS < 0
    or not 0 <= SHOP_FAILURE_RATE <= 1
    or SHOP_TIMEOUT_SECONDS <= 0
):
    raise ValueError(
        "SHOP_DELAY_MS must be non-negative, SHOP_FAILURE_RATE in [0, 1], "
        "and SHOP_TIMEOUT_SECONDS positive."
    )

_ORDER_ID = {"type": "string", "pattern": "^[A-Za-z0-9-]{1,32}$"}
_SKU = {"type": "string", "pattern": "^[A-Z0-9][A-Z0-9-]{1,19}$"}
_PRICE = {
    "type": "string",
    "pattern": "^(?:0|[1-9][0-9]{0,7})(?:\\.[0-9]{1,2})?$",
}
_ITEMS = {
    "type": "array",
    "minItems": 1,
    "maxItems": 30,
    "items": {
        "type": "object",
        "properties": {
            "sku": _SKU,
            "quantity": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["sku", "quantity"],
        "additionalProperties": False,
    },
}


def _function(
    name: str, description: str, properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS_SCHEMA = [
    _function(
        "list_orders",
        "Показати замовлення поточного клієнта. Не приймає ідентифікатор клієнта.",
        {},
    ),
    _function(
        "get_order",
        "Переглянути статус і безпечні деталі одного замовлення клієнта.",
        {"order_id": _ORDER_ID},
        ["order_id"],
    ),
    _function(
        "search_catalog",
        "Спочатку викликай цей інструмент для пошуку товару за назвою чи описом. "
        "Для наступних інструментів використовуй лише повернений ним SKU.",
        {
            "query": {"type": "string", "minLength": 1, "maxLength": 120},
            "category": {"type": "string", "minLength": 1, "maxLength": 60},
            "max_price": _PRICE,
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ["query"],
    ),
    _function(
        "get_product",
        "Переглянути безпечні характеристики товару за артикулом.",
        {"sku": _SKU},
        ["sku"],
    ),
    _function(
        "check_stock",
        "Перевірити наявність за точним SKU, отриманим із search_catalog. "
        "Не вигадуй SKU.",
        {"sku": _SKU},
        ["sku"],
    ),
    _function(
        "calculate_delivery",
        "Розрахувати ціну й строк доставки товарів із каталогу. Використовуй "
        "лише SKU, отримані через search_catalog.",
        {
            "city": {"type": "string", "minLength": 1, "maxLength": 80},
            "method": {"type": "string", "enum": list(service.DELIVERY_METHODS)},
            "items": _ITEMS,
        },
        ["city", "method", "items"],
    ),
    _function(
        "request_return",
        "Створити заявку на повернення товару з власного замовлення. "
        "Викликай лише на пряме прохання клієнта.",
        {
            "order_id": _ORDER_ID,
            "sku": _SKU,
            "reason": {"type": "string", "enum": sorted(service.RETURN_REASONS)},
            "quantity": {"type": "integer", "minimum": 1, "maximum": 1000},
            "comment": {"type": "string", "maxLength": 500},
        },
        ["order_id", "sku", "reason", "quantity"],
    ),
]

_TOOL_SCHEMAS = {
    item["function"]["name"]: item["function"]["parameters"]
    for item in TOOLS_SCHEMA
}
_VALIDATORS = {
    name: Draft202012Validator(schema) for name, schema in _TOOL_SCHEMAS.items()
}


@dataclass
class ToolResult:
    name: str
    arguments: Any
    status: str
    reason: str | None
    result: Any
    content: str
    action: str
    elapsed: float = 0.0


def specs() -> list[dict[str, Any]]:
    """Return the same allow-listed tool descriptions sent to the model."""
    return deepcopy(TOOLS_SCHEMA)


def _json_result(result: Any) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _error(name: str, arguments: Any, code: str, message: str) -> ToolResult:
    result = {"error": message, "code": code}
    return ToolResult(
        name=name,
        arguments=arguments,
        status="error" if code in {"unavailable", "timeout"} else "rejected",
        reason=message,
        result=result,
        content=_json_result(result),
        action="not executed",
    )


def _public_order(order: dict[str, Any]) -> dict[str, Any]:
    delivery = order.get("delivery") or {}
    return {
        "order_id": order.get("order_id"),
        "created_at": order.get("created_at"),
        "status": order.get("status"),
        "status_label": order.get("status_label"),
        "total": order.get("total"),
        "items": [
            {
                key: item.get(key)
                for key in ("sku", "name", "quantity", "price")
                if key in item
            }
            for item in order.get("items", [])
        ],
        "delivery": {
            key: delivery.get(key)
            for key in (
                "method", "carrier", "city", "point", "tracking",
                "delivered_at", "cost",
            )
            if key in delivery
        },
    }


def _public_product(product: dict[str, Any]) -> dict[str, Any]:
    return {
        key: product[key]
        for key in (
            "sku", "name", "category", "price", "weight_kg",
            "warranty_months", "returnable", "bulky",
        )
        if key in product
    }


def _run(name: str, args: dict[str, Any], customer_id: str) -> tuple[Any, str]:
    if name == "list_orders":
        orders = service.list_orders(customer_id)
        return (
            [
                {
                    key: order.get(key)
                    for key in (
                        "order_id", "created_at", "status", "status_label",
                        "total", "items_count",
                    )
                }
                for order in orders
            ],
            "listed only orders belonging to the session customer",
        )

    if name == "get_order":
        order = service.get_order(args["order_id"])
        if order.get("customer_id") != customer_id:
            raise service.NotFound("замовлення не знайдено")
        return _public_order(order), "verified order ownership and filtered private fields"

    if name == "search_catalog":
        max_price = args.get("max_price")
        if max_price is not None:
            try:
                if not Decimal(max_price).is_finite():
                    raise InvalidOperation
            except InvalidOperation as exc:
                raise ValueError("max_price має бути коректною сумою") from exc
        products = service.search_products(
            query=args["query"],
            category=args.get("category"),
            max_price=max_price,
            limit=args.get("limit", 10),
        )
        return (
            [
                {
                    key: product.get(key)
                    for key in ("sku", "name", "category", "price")
                }
                for product in products
            ],
            "filtered catalog fields",
        )

    if name == "get_product":
        return _public_product(service.get_product(args["sku"])), "filtered product fields"

    if name == "check_stock":
        stock = service.get_stock(args["sku"])
        return (
            {
                key: stock.get(key)
                for key in ("sku", "available", "expected_restock")
            },
            "filtered stock fields",
        )

    if name == "calculate_delivery":
        quote = service.delivery_quote(
            args["city"], args["method"], args["items"]
        )
        return (
            {
                key: quote.get(key)
                for key in (
                    "city", "method", "order_total", "weight_kg",
                    "cost", "days", "note",
                )
            },
            "calculated delivery for validated catalog items",
        )

    if name == "request_return":
        order = service.get_order(args["order_id"])
        if order.get("customer_id") != customer_id:
            raise service.NotFound("замовлення не знайдено")
        request = service.create_return(
            order_id=args["order_id"],
            sku=args["sku"],
            reason=args["reason"],
            quantity=args["quantity"],
            comment=args.get("comment", ""),
        )
        return (
            {
                key: request.get(key)
                for key in (
                    "return_id", "order_id", "sku", "quantity", "reason",
                    "created_at", "status", "next_steps",
                )
            },
            "verified order ownership and created return request",
        )

    raise AssertionError("Tool name must come from the server-side allow-list")


def call(tool_name: str, raw_arguments: str, customer_id: str) -> ToolResult:
    """Validate an untrusted model call, execute an allow-listed tool and filter output."""
    if tool_name not in _VALIDATORS:
        return _error(tool_name, raw_arguments, "unknown_tool", "інструмент недоступний")
    if not isinstance(raw_arguments, str) or len(raw_arguments) > 8192:
        return _error(
            tool_name, "[arguments omitted: over 8192 characters]", "invalid_arguments",
            "аргументи відсутні або перевищують дозволений розмір",
        )
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        return _error(
            tool_name, raw_arguments, "invalid_arguments",
            "аргументи інструмента мають бути коректним JSON",
        )
    if not isinstance(arguments, dict):
        return _error(
            tool_name, arguments, "invalid_arguments",
            "аргументи інструмента мають бути JSON-об'єктом",
        )
    schema = _TOOL_SCHEMAS[tool_name]
    required = set(schema.get("required", []))
    arguments = {
        key: value
        for key, value in arguments.items()
        if value is not None or key in required
    }
    errors = sorted(
        _VALIDATORS[tool_name].iter_errors(arguments),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        first = errors[0]
        field = ".".join(str(part) for part in first.absolute_path) or "arguments"
        return _error(
            tool_name,
            arguments,
            "invalid_arguments",
            f"некоректний параметр {field}: {first.message}",
        )
    if not isinstance(customer_id, str) or not re.fullmatch(
        r"C-[0-9]{4,12}", customer_id
    ):
        return _error(
            tool_name, arguments, "invalid_session", "не визначено клієнта сесії"
        )

    if SHOP_DELAY_MS > SHOP_TIMEOUT_SECONDS * 1000:
        return _error(
            tool_name,
            arguments,
            "timeout",
            "сервіс магазину не відповів у відведений час; спробуйте пізніше",
        )

    started = time.perf_counter()
    try:
        result, action = _run(tool_name, arguments, customer_id)
    except service.ShopError as exc:
        code = getattr(exc, "code", "shop_error")
        return ToolResult(
            name=tool_name,
            arguments=arguments,
            status="error" if code == "unavailable" else "rejected",
            reason=str(exc),
            result={"error": str(exc), "code": code},
            content=_json_result({"error": str(exc), "code": code}),
            action="shop service returned an error",
            elapsed=time.perf_counter() - started,
        )
    except ValueError as exc:
        return _error(tool_name, arguments, "invalid_arguments", str(exc))

    content = _json_result(result)
    return ToolResult(
        name=tool_name,
        arguments=arguments,
        status="ok",
        reason=None,
        result=result,
        content=content,
        action=action,
        elapsed=time.perf_counter() - started,
    )


def execute_tool(tool_name: str, args: dict[str, Any], customer_id: str) -> str:
    """Compatibility wrapper; prefer `call` for traceable execution."""
    return call(tool_name, _json_result(args), customer_id).content
