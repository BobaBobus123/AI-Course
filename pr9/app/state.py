from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class WorkflowState(BaseModel):
    run_id: str
    customer_id: str
    text: str
    status: str = "running"  # running, waiting_approval, completed, rejected, stopped
    current_step: str = "parse"
    parsed_data: Optional[Dict[str, Any]] = None
    order_info: Optional[Dict[str, Any]] = None
    journal: List[Dict[str, Any]] = Field(default_factory=list)
    tokens_used: int = 0
    start_time: float = 0.0
    answer: Optional[str] = None
    operator_comment: Optional[str] = None
    created_at: str = ""