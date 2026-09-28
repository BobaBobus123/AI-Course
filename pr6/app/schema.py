from typing import List, Optional
from pydantic import BaseModel, Field

class SourceReference(BaseModel):
    source_id: int = Field(description="Номер джерела у форматі [із контексту]")
    document_name: str = Field(description="Назва документа")
    section: str = Field(description="Розділ")

class RAGResponseSchema(BaseModel):
    response_text: str = Field(description="Ввічлива відповідь клієнту українською мовою з опорою виключно на контекст.")
    found_in_base: bool = Field(description="True, якщо відповідь повністю знайдено у фрагментах бази знань; False, якщо у базі немає інформації.")
    cited_source_ids: List[int] = Field(default=[], description="Список номерів фрагментів, на які спирається відповідь.")