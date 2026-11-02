from typing import Literal, Optional
from pydantic import BaseModel, Field

class ParsedRequest(BaseModel):
    intent: Literal["return", "cancel", "status", "other"] = Field(
        description="Вид звернення: return (повернення), cancel (скасування), status (стан замовлення), other (інше)"
    )
    order_id: Optional[str] = Field(
        default=None, description="Номер замовлення, якщо вказано"
    )
    sku: Optional[str] = Field(
        default=None, description="Артикул товару (SKU), якщо вказано"
    )
    reason: Optional[str] = Field(
        default=None, description="Причина повернення або скасування"
    )
    quantity: Optional[int] = Field(
        default=1, description="Кількість товару"
    )