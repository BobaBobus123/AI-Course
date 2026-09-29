from pydantic import BaseModel, Field
from typing import List, Optional

class CompanyInfo(BaseModel):
    name: Optional[str] = Field(None, description="Назва компанії")
    code: Optional[str] = Field(None, description="Код ЄДРПОУ або РНОКПП (8 або 10 цифр). null, якщо немає.")
    iban: Optional[str] = Field(None, description="Рахунок IBAN. null, якщо немає.")

class LineItem(BaseModel):
    name: Optional[str] = Field(None, description="Найменування товару/послуги")
    unit: Optional[str] = Field(None, description="Одиниця виміру")
    quantity: Optional[str] = Field(None, description="Кількість (рядок, напр. '2' або '1.5')")
    price: Optional[str] = Field(None, description="Ціна за одиницю (рядок, напр. '150.00')")
    sum: Optional[str] = Field(None, description="Сума за позицією (рядок)")

class InvoiceSchema(BaseModel):
    doc_type: Optional[str] = Field(None, description="Тип документа (наприклад, 'рахунок на оплату', 'видаткова накладна')")
    number: Optional[str] = Field(None, description="Номер документа. null, якщо немає.")
    date: Optional[str] = Field(None, description="Дата документа. null, якщо немає.")
    due_date: Optional[str] = Field(None, description="Строк дії або дата оплати. null, якщо немає.")
    
    supplier: CompanyInfo = Field(..., description="Постачальник")
    buyer: CompanyInfo = Field(..., description="Покупець")
    
    items: List[LineItem] = Field(default_factory=list, description="Позиції в документі")
    
    total_no_vat: Optional[str] = Field(None, description="Разом без ПДВ (рядок). null, якщо немає.")
    vat: Optional[str] = Field(None, description="ПДВ (рядок). null, якщо немає.")
    total_to_pay: Optional[str] = Field(None, description="Всього до сплати (рядок)")