from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

from shop import service
from . import assistant, workflow, store, tools

app = FastAPI(title="Помічник клієнта — ПР9")
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

class AskRequest(BaseModel):
    customer_id: str
    question: str = ""
    text: str = ""
    mode: str = "workflow"

class OperatorDecisionRequest(BaseModel):
    action: str = "approve"
    comment: str = ""

@app.get("/", response_class=HTMLResponse)
def get_index(request: Request):
    try:
        raw_runs = store.list_runs() or []
        safe_runs = [r if isinstance(r, dict) else r.model_dump() for r in raw_runs if r]
    except Exception:
        safe_runs = []

    try:
        raw_customers = service.list_customers() or []
        safe_customers = [c if isinstance(c, dict) else dict(c) for c in raw_customers if c]
    except Exception:
        safe_customers = []

    template = templates.env.get_template("index.html")
    html_content = template.render(
        request=request,
        runs=safe_runs,
        customers=safe_customers,
        tools_schema=tools.TOOLS_SCHEMA
    )
    return HTMLResponse(content=html_content)

@app.get("/api/customers")
def api_customers():
    return service.list_customers()

@app.get("/api/runs")
def api_runs(status: Optional[str] = Query(None)):
    runs = store.list_runs() or []
    safe_runs = [r if isinstance(r, dict) else r.model_dump() for r in runs if r]
    
    if status:
        filtered = []
        for r in safe_runs:
            st = r.get("status")
            if status in ["awaiting_approval", "waiting_approval"] and st in ["waiting_approval", "awaiting_approval", "stopped"]:
                filtered.append(r)
            elif st == status:
                filtered.append(r)
        return filtered
    return safe_runs

@app.get("/api/runs/{run_id}")
def api_get_run(run_id: str):
    st = store.load_state(run_id)
    if not st:
        return {"error": "Run not found"}
    return st if isinstance(st, dict) else st.model_dump()

@app.post("/api/runs/{run_id}/decision")
def api_run_decision(run_id: str, payload: OperatorDecisionRequest):
    state = workflow.handle_operator_action(run_id, payload.action, payload.comment)
    return state

@app.post("/api/workflow/{run_id}/operator")
def api_operator_action(run_id: str, payload: OperatorDecisionRequest):
    return workflow.handle_operator_action(run_id, payload.action, payload.comment)

def _start_process(payload: AskRequest):
    query = payload.question or payload.text
    if payload.mode == "agent":
        res = assistant.handle_customer_query(payload.customer_id, query)
        return {
            "run_id": "agent-mode",
            "status": "completed",
            "answer": res.get("answer"),
            "journal": [{"step": "agent", "type": "model", "status": "ok", "message": "Виконано через агента ПР8", "data": res, "time": ""}]
        }
    else:
        state = workflow.create_run(payload.customer_id, query)
        return state

@app.post("/api/requests")
def api_requests(payload: AskRequest):
    return _start_process(payload)

@app.post("/api/workflow/start")
def api_workflow_start(payload: AskRequest):
    return _start_process(payload)

@app.post("/api/reset")
def api_reset():
    service.reset()
    return {"ok": True}