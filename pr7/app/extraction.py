import time
from dataclasses import dataclass, field
from typing import Any

from .images import prepare_image
from .llm import extract_invoice_data
from .rules import validate_invoice

@dataclass
class Result:
    """Результат конвеєра — те, з чим працює веб-рівень."""
    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[dict] = field(default_factory=list) # Замінили Issue на dict для сумісності з rules.py
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None

def decide(document: dict, issues: list[dict]) -> tuple[str, list[str]]:
    """Вирішити долю документа за вилученими полями й знайденими проблемами."""
    decision = "auto"
    reasons = []
    
    # 1. Reject (Відхилити)
    doc_type = str(document.get("doc_type", "")).lower()
    if not doc_type or "рахунок" not in doc_type:
        return "reject", ["Документ не розпізнано як рахунок на оплату (можливо, це видаткова накладна)."]
        
    # 2. Review (На перевірку людині) - якщо є проблеми з математикою або форматами
    if issues:
        decision = "review"
        reasons.extend([issue["message"] for issue in issues])
        
    # 3. Review (На перевірку людині) - якщо бракує критичних полів
    supplier = document.get("supplier", {})
    if not supplier.get("iban") or not document.get("total_to_pay"):
        decision = "review"
        reasons.append("Відсутні критичні поля (IBAN або підсумкова сума).")
        
    return decision, reasons

def process(content: bytes) -> Result:
    """Обробити файл: підготувати → вилучити → перевірити → вирішити."""
    elapsed = {}
    
    # 1. Підготовка зображення
    t0 = time.time()
    try:
        data_url, img_meta = prepare_image(content)
    except Exception as e:
        return Result(decision="reject", reasons=[f"Помилка зображення: {e}"])
    elapsed["prepare"] = round(time.time() - t0, 3)
    
    # 2. Модель (LLM)
    t0 = time.time()
    try:
        extracted_data_obj, llm_meta = extract_invoice_data(data_url)
        # Перетворюємо Pydantic-об'єкт на звичайний словник
        extracted_data_dict = extracted_data_obj.model_dump() if hasattr(extracted_data_obj, 'model_dump') else extracted_data_obj.dict()
    except Exception as e:
        return Result(decision="reject", reasons=[f"Помилка моделі: {e}"])
    elapsed["extraction"] = round(time.time() - t0, 3)
    
    # 3. Програмні правила
    t0 = time.time()
    # rules.py очікує Pydantic-об'єкт
    issues = validate_invoice(extracted_data_obj)
    elapsed["checks"] = round(time.time() - t0, 3)
    
    # 4. Ухвалення рішення
    decision, reasons = decide(extracted_data_dict, issues)
    
    return Result(
        decision=decision,
        reasons=reasons,
        document=extracted_data_dict,
        issues=issues,
        image=img_meta,
        model="gemini-2.5-flash",
        elapsed=elapsed,
        usage=llm_meta
    )