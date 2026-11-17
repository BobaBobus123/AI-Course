"""Pure matching and policy rules used by the customer workflow."""

import re
from datetime import datetime, timedelta

from shop import service


def _normal(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower(), flags=re.UNICODE).strip()


def detect_multiple(text: str) -> bool:
    normalized = _normal(text)
    asks_cancel = bool(re.search(r"\bскасу\w*|\bвідмін\w*", normalized))
    asks_return = bool(re.search(r"\bповерн\w*|\bзаявк\w*", normalized))
    return asks_cancel and asks_return


def reason_from_text(text: str, parsed_reason: str | None) -> str | None:
    normalized = _normal(text)
    if any(term in normalized for term in (
        "не працю", "злам", "зламав", "перестав", "не заряд", "дефект",
        "пошкод", "розбит",
    )):
        return "damaged" if "пошкод" in normalized or "розбит" in normalized else "defect"
    if any(term in normalized for term in ("не підійш", "завел", "замал", "не сподоб")):
        return "not_suitable"
    return parsed_reason


def match_item(order: dict, item: str | None) -> dict | None:
    items = order.get("items", [])
    if not item or not item.strip():
        return items[0] if len(items) == 1 else None
    query = _normal(item)
    exact = [entry for entry in items
             if query == _normal(entry["sku"]) or query in _normal(entry["name"])]
    if len(exact) == 1:
        return exact[0]
    query_words = {word for word in query.split() if len(word) > 2}
    ranked = []
    for entry in items:
        words = set(_normal(entry["name"] + " " + entry["sku"]).split())
        score = len(query_words & words)
        if score:
            ranked.append((score, entry))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
        return ranked[0][1]
    return None


def return_policy_problem(order: dict, product: dict, reason: str) -> str | None:
    if order["status"] != "delivered":
        return "Повернення оформлюється лише для отриманого замовлення."
    delivered = datetime.fromisoformat(order["delivery"]["delivered_at"]).date()
    age = (service.today() - delivered).days
    if reason == "not_suitable":
        if not product.get("returnable"):
            return "Цей товар належної якості не підлягає поверненню."
        if age > service.RETURN_WINDOW_DAYS:
            return f"Строк повернення належної якості минув: від отримання пройшло {age} днів."
    if reason == "defect":
        months = int(product.get("warranty_months", 0))
        if service.today() > delivered + timedelta(days=30 * months):
            return "Гарантійний строк минув."
    return None
