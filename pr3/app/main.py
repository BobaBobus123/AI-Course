from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.inference import generate_support_response
import uvicorn

app = FastAPI(title="E-commerce Support LLM Assistant API", version="1.0")

class SupportRequest(BaseModel):
    query: str = Field(..., description="Звернення клієнта до служби підтримки", min_length=2)
    temperature: float = Field(0.3, description="Параметр генерації (температура)", ge=0.0, le=2.0)

@app.post("/support/")
async def support_endpoint(payload: SupportRequest):

    result = generate_support_response(payload.query, payload.temperature)
    return result

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)