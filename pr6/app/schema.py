from pydantic import BaseModel, ConfigDict, Field


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: int = Field(description="Номер джерела у форматі [із контексту]")
    document_name: str = Field(description="Назва документа")
    section: str = Field(description="Розділ")
    updated: str = Field(description="Дата редакції документа")


class RAGResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_text: str = Field(
        min_length=1,
        description=(
            "Ввічлива відповідь клієнту українською мовою з опорою виключно "
            "на контекст і посиланнями [номер]."
        ),
    )
    found_in_base: bool = Field(
        description=(
            "True, якщо відповідь підтверджена фрагментами бази знань; "
            "False, якщо інформації немає."
        )
    )
    cited_source_ids: list[int] = Field(
        default_factory=list,
        description="Номери використаних фрагментів із наданого контексту.",
    )