import re
import json
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator

ORDER_REGEX = re.compile(r"^ORD-\d{5}$")

TopicLiteral = Literal["доставка", "оплата", "повернення", "гарантія", "інше"]

DEPARTMENT_MAPPING: Dict[str, str] = {
    "доставка": "Відділ логістики",
    "оплата": "Фінансовий відділ",
    "повернення": "Відділ повернень та рекламацій",
    "гарантія": "Сервісний центр",
    "інше": "Загальна лінія підтримки",
}


class ModelRawOutputSchema(BaseModel):
    """
    Схема відповіді LLM (використовується для генерації JSON Schema та первинної перевірки).
    Забороняє зайві поля (extra='forbid') і перевіряє формат номера замовлення.
    """
    model_config = ConfigDict(extra="forbid")

    response_text: str = Field(
        min_length=5,
        max_length=1500,
        description="Ввічлива, чітка та лаконічна відповідь клієнту українською мовою."
    )
    topic: TopicLiteral = Field(
        description="Тема звернення клієнта з фіксованого переліку."
    )
    is_based_on_rules: bool = Field(
        description=(
            "True, якщо відповідь повністю ґрунтується на правилах магазину з context.md. "
            "False, якщо правила про це мовчать або питання поза межами правил."
        )
    )
    needs_clarification: bool = Field(
        description=(
            "True, якщо звернення неоднозначне або бракує даних для вирішення питання "
            "(наприклад, номера замовлення ORD-XXXXX чи опису проблеми)."
        )
    )
    customer_requests_human: bool = Field(
        description=(
            "True, якщо клієнт прямо просить з'єднати з оператором, висловлює претензію "
            "або ситуація вимагає втручання людини."
        )
    )
    order_number: Optional[str] = Field(
        default=None,
        description=(
            "Номер замовлення у форматі ORD-XXXXX (де XXXXX — 5 цифр), "
            "якщо клієнт назвав його в поточному або попередніх повідомленнях; інакше null."
        )
    )

    @field_validator("order_number", mode="before")
    @classmethod
    def validate_and_clean_order_number(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Поле order_number має бути рядком або null.")
        cleaned = value.strip()
        if cleaned.casefold() in ("", "null", "немає", "не вказано"):
            return None
        if not ORDER_REGEX.fullmatch(cleaned):
            raise ValueError(
                f"Невалідний формат номера замовлення '{value}'. Очікується ORD-XXXXX (рівно 5 цифр)."
            )
        return cleaned


class ValidatedSupportCard(BaseModel):
    """
    Перевірена картка звернення, яку отримує веб-рівень застосунку.
    """
    response_text: str
    topic: TopicLiteral
    target_department: str
    is_based_on_rules: bool
    needs_clarification: bool
    customer_requests_human: bool
    escalate_to_operator: bool
    order_number: Optional[str] = None


def build_validated_card(raw_data: str | dict | ModelRawOutputSchema) -> ValidatedSupportCard:
    """
    Шар програмної перевірки між моделлю та веб-рівнем:
    1. Валідує синтаксис і типи за JSON Schema (ModelRawOutputSchema).
    2. Програмно (в коді, а не в моделі) визначає цільовий відділ target_department
       та остаточне рішення escalate_to_operator.
    """
    if isinstance(raw_data, str):
        cleaned_str = raw_data.strip()
        if cleaned_str.startswith("```"):
            cleaned_str = re.sub(r"^```(?:json)?\s*", "", cleaned_str)
            cleaned_str = re.sub(r"\s*```$", "", cleaned_str)
        parsed_dict = json.loads(cleaned_str)
        validated_raw = ModelRawOutputSchema.model_validate(parsed_dict)
    elif isinstance(raw_data, dict):
        validated_raw = ModelRawOutputSchema.model_validate(raw_data)
    elif isinstance(raw_data, ModelRawOutputSchema):
        validated_raw = raw_data
    else:
        raise ValueError(f"Непідтримуваний тип даних для валідації: {type(raw_data)}")

    should_escalate = (
        validated_raw.customer_requests_human
        or (not validated_raw.is_based_on_rules and not validated_raw.needs_clarification)
        or (
            validated_raw.topic in ("повернення", "гарантія")
            and validated_raw.order_number is not None
            and not validated_raw.needs_clarification
        )
    )

    department = DEPARTMENT_MAPPING.get(validated_raw.topic, "Загальна лінія підтримки")

    return ValidatedSupportCard(
        response_text=validated_raw.response_text,
        topic=validated_raw.topic,
        target_department=department,
        is_based_on_rules=validated_raw.is_based_on_rules,
        needs_clarification=validated_raw.needs_clarification,
        customer_requests_human=validated_raw.customer_requests_human,
        escalate_to_operator=should_escalate,
        order_number=validated_raw.order_number,
    )


def create_safe_fallback_card(known_order_number: Optional[str] = None) -> ValidatedSupportCard:
    """
    Безпечна відповідь на випадок, якщо модель повернула невалідні дані після всіх спроб.
    """
    if known_order_number is not None and (
        not isinstance(known_order_number, str)
        or not ORDER_REGEX.fullmatch(known_order_number)
    ):
        known_order_number = None

    return ValidatedSupportCard(
        response_text=(
            "Перепрошую, під час автоматичної обробки вашого звернення виникла технічна затримка. "
            "Я передаю ваш діалог оператору служби підтримки для швидкого вирішення питання."
        ),
        topic="інше",
        target_department="Загальна лінія підтримки",
        is_based_on_rules=False,
        needs_clarification=False,
        customer_requests_human=False,
        escalate_to_operator=True,
        order_number=known_order_number,
    )


def get_json_schema_dict() -> Dict[str, Any]:
    """Повертає JSON Schema у вигляді словника."""
    return ModelRawOutputSchema.model_json_schema()