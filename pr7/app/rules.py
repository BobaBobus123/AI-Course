import json
import os
from app.schema import InvoiceSchema

def load_suppliers():
    """Завантажує довідник постачальників із файлу JSON."""
    # Отримуємо шлях до папки reference на рівень вище від папки app
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "reference", "suppliers.json")
    
    if not os.path.exists(filepath):
        print(f"Попередження: Довідник постачальників не знайдено за шляхом {filepath}")
        return []
        
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_invoice(invoice: InvoiceSchema) -> list[str]:
    issues = []
    
    # 1. ПЕРЕВІРКА МАТЕМАТИКИ (Рядки)
    calculated_total_no_vat = 0
    if invoice.items:
        for item in invoice.items:
            if item.quantity and item.price and item.sum:
                try:
                    q = float(item.quantity)
                    p = float(item.price)
                    s = float(item.sum)
                    # Округлюємо до 2 знаків, щоб уникнути проблем із плаваючою комою
                    if round(q * p, 2) != round(s, 2):
                        issues.append(f"Арифметика рядка '{item.name}': {q} * {p} != {s}")
                    calculated_total_no_vat += s
                except ValueError:
                    issues.append(f"Невірний формат чисел у рядку '{item.name}'")
                    
    # 2. ПЕРЕВІРКА АРИФМЕТИКИ (Підсумок)
    if invoice.total_no_vat is not None:
        try:
            total_no_vat = float(invoice.total_no_vat)
            if round(calculated_total_no_vat, 2) != round(total_no_vat, 2):
                issues.append(f"Сума без ПДВ не сходиться: розраховано {calculated_total_no_vat:.2f}, вказано {total_no_vat:.2f}")
        except ValueError:
            pass

    # 3. ПЕРЕВІРКА ПОКУПЦЯ (Має бути наша компанія "Сузір'я Рітейл")
    MY_COMPANY_CODE = "44172914"
    if invoice.buyer:
        if invoice.buyer.code and invoice.buyer.code != MY_COMPANY_CODE:
            issues.append(f"Рахунок виставлено на чужу компанію! ЄДРПОУ покупця: {invoice.buyer.code}")
    else:
        issues.append("Не вказано покупця в рахунку.")

    # 4. ПЕРЕВІРКА ПОСТАЧАЛЬНИКА ЗА ДОВІДНИКОМ
    suppliers_db = load_suppliers()
    if invoice.supplier and invoice.supplier.code:
        supplier_code = invoice.supplier.code
        # Прибираємо всі пробіли з IBAN для точного порівняння
        supplier_iban = invoice.supplier.iban.replace(" ", "") if invoice.supplier.iban else None
        
        # Шукаємо постачальника в JSON базі за кодом ЄДРПОУ
        matched_supplier = next((s for s in suppliers_db if s["code"] == supplier_code), None)
        
        if matched_supplier:
            expected_iban = matched_supplier["iban"].replace(" ", "")
            if supplier_iban and supplier_iban != expected_iban:
                issues.append(f"Увага! IBAN постачальника не збігається з довідником. Вказано: {supplier_iban}, очікується: {expected_iban}")
        else:
            issues.append(f"Постачальника з кодом {supplier_code} немає в базі перевірених контрагентів (suppliers.json).")
    else:
        issues.append("Не вдалося розпізнати код ЄДРПОУ постачальника для перевірки за довідником.")

    return issues