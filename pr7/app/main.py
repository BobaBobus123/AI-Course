import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.images import prepare_image
from app.llm import extract_invoice_data
from app.rules import validate_invoice

app = FastAPI(title="Мультимодальний аналіз рахунків (ПР7)")

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    # Оновлений синтаксис для нових версій FastAPI / Starlette
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/process")
async def process_document(file: UploadFile = File(...)):
    """Головний конвеєр обробки документа."""
    start_total = time.time()
    
    
    t0 = time.time()
    try:
        contents = await file.read()
        data_url, img_meta = prepare_image(contents)
    except Exception as e:
        
        return {"decision": "reject", "reasons": [str(e)], "data": None, "meta": {}}
    t_prep = time.time() - t0

    
    t0 = time.time()
    try:
        extracted_data, llm_meta = extract_invoice_data(data_url)
    except Exception as e:
        # Якщо API моделі впало (ліміти, таймаут) - reject
        return {"decision": "reject", "reasons": [f"Помилка моделі: {str(e)}"], "data": None, "meta": {}}
    t_llm = time.time() - t0

    
    t0 = time.time()
    issues = validate_invoice(extracted_data)
    t_rules = time.time() - t0
    
    
    decision = "auto"
    reasons = []
    
    
    if not extracted_data.doc_type or "рахунок" not in extracted_data.doc_type.lower():
        decision = "reject"
        reasons.append("Документ не розпізнано як рахунок на оплату (можливо, це видаткова накладна).")
    
    
    elif issues:
        decision = "review"
        reasons.extend([issue["message"] for issue in issues])
    elif not extracted_data.supplier.iban or not extracted_data.total_to_pay:
        decision = "review"
        reasons.append("Відсутні критичні поля (IBAN або підсумкова сума).")
        
    

    total_time = time.time() - start_total
    
    
    meta = {
        "image_size_orig": img_meta["original_size"],
        "image_size_new": img_meta["new_size"],
        "time_prep": round(t_prep, 2),
        "time_llm": round(t_llm, 2),
        "time_rules": round(t_rules, 3),
        "time_total": round(total_time, 2),
        "tokens": llm_meta
    }

    return {
        "decision": decision,
        "reasons": reasons,
        "issues": issues,
        "data": extracted_data.dict(),
        "meta": meta
    }