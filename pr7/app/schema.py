from pydantic import BaseModel, ConfigDict, Field


class CompanyInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    name: str | None = Field(
        default=None,
        description="Назва сторони точно як у документі; null, якщо відсутня.",
    )
    code: str | None = Field(
        default=None,
        description="ЄДРПОУ або РНОКПП точно як у документі; null, якщо відсутній.",
    )
    iban: str | None = Field(
        default=None,
        description="IBAN точно як у документі; null, якщо відсутній.",
    )


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    name: str | None = Field(default=None, description="Назва позиції.")
    unit: str | None = Field(default=None, description="Одиниця виміру.")
    quantity: str | None = Field(default=None, description="Кількість.")
    price: str | None = Field(default=None, description="Ціна за одиницю.")
    sum: str | None = Field(default=None, description="Надрукована сума позиції.")


class InvoiceSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    doc_type: str | None = Field(default=None, description="Тип документа.")
    number: str | None = Field(default=None, description="Номер рахунку.")
    date: str | None = Field(default=None, description="Дата рахунку.")
    due_date: str | None = Field(default=None, description="Строк оплати, якщо є.")
    supplier: CompanyInfo = Field(
        default_factory=CompanyInfo, description="Постачальник."
    )
    buyer: CompanyInfo = Field(default_factory=CompanyInfo, description="Покупець.")
    items: list[LineItem] = Field(
        default_factory=list,
        description="Усі надруковані позиції; нічого не обчислювати самостійно.",
    )
    total_no_vat: str | None = Field(
        default=None, description="Надрукована сума без ПДВ."
    )
    vat: str | None = Field(default=None, description="Надрукована сума ПДВ.")
    total_to_pay: str | None = Field(
        default=None, description="Надрукована сума до сплати."
    )
