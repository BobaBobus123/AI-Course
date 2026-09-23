from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from app.inference import generate_dialog_response

app = FastAPI(title="E-commerce Context-Aware Support Assistant API (PR4)")

conversation_history: List[dict] = []

class ChatRequest(BaseModel):
    query: str
    temperature: Optional[float] = 0.3
    mode: Optional[str] = "context_engineering"

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <title>ПР4: Діалоговий асистент зі структурованим виводом</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f7f6; margin: 0; padding: 20px; }
            .container { max-width: 900px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; font-size: 22px; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
            .chat-box { height: 350px; border: 1px solid #ddd; border-radius: 5px; padding: 15px; overflow-y: scroll; background: #fafafa; margin-bottom: 15px; }
            .message { margin-bottom: 10px; padding: 8px 12px; border-radius: 6px; max-width: 80%; }
            .user { background: #dcf8c6; margin-left: auto; text-align: right; }
            .assistant { background: #e2eef8; margin-right: auto; }
            .controls { display: flex; gap: 10px; margin-bottom: 15px; }
            input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .card-panel { background: #fff3cd; border: 1px solid #ffeeba; padding: 10px; border-radius: 5px; margin-top: 15px; font-size: 13px; }
            .meta { font-size: 11px; color: #666; margin-top: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Діалоговий AI-асистент підтримки (ПР4)</h1>
            <div class="chat-box" id="chatBox">
                <div class="message assistant">Вітаю! Чим можу допомогти вам у нашому інтернет-магазині?</div>
            </div>
            
            <div class="controls">
                <input type="text" id="queryInput" placeholder="Введіть звернення (наприклад: 'Мене звати Іван, замовлення ORD-12345')..." onkeydown="if(event.key==='Enter') sendQuery()">
                <button onclick="sendQuery()">Надіслати</button>
            </div>

            <div class="card-panel" id="cardPanel">
                <strong>Картка звернення (Structured Output):</strong>
                <pre id="cardOutput">Поки немає даних звернення.</pre>
            </div>
        </div>

        <script>
            async function sendQuery() {
                const input = document.getElementById('queryInput');
                const chatBox = document.getElementById('chatBox');
                const cardOutput = document.getElementById('cardOutput');
                const query = input.value.trim();
                
                if (!query) return;

                chatBox.innerHTML += `<div class="message user">${query}</div>`;
                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query, temperature: 0.3, mode: 'context_engineering' })
                    });
                    
                    const data = await response.json();
                    if (response.ok) {
                        const parsed = data.parsed_data;
                        chatBox.innerHTML += `<div class="message assistant">${parsed.response_text}<div class="meta">Тема: ${parsed.topic} | Час: ${data.execution_time_seconds}с | Токени: ${data.tokens.total_tokens}</div></div>`;
                        cardOutput.textContent = JSON.stringify(data, null, 2);
                    } else {
                        chatBox.innerHTML += `<div class="message assistant" style="background: #f8d7da;">Помилка: ${data.detail}</div>`;
                    }
                } catch (err) {
                    chatBox.innerHTML += `<div class="message assistant" style="background: #f8d7da;">Помилка з'єднання з сервером.</div>`;
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    global conversation_history
    
    conversation_history.append({"role": "user", "content": payload.query})
    
    result = generate_dialog_response(
        history=conversation_history, 
        user_query=payload.query, 
        temperature=payload.temperature, 
        mode=payload.mode
    )
    
    assistant_text = result["parsed_data"]["response_text"]
    conversation_history.append({"role": "assistant", "content": assistant_text})
    
    return result

@app.post("/reset")
async def reset_history():
    global conversation_history
    conversation_history = []
    return {"status": "success", "message": "Історію діалогу очищено."}