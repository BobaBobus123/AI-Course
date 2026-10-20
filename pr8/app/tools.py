import json
from shop import service

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Отримати інформацію про замовлення.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Пошук товару в каталозі.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Перевірити склад за артикулом.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"}
                },
                "required": ["sku"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_delivery",
            "description": "Розрахувати доставку.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "method": {"type": "string", "enum": ["courier", "post", "pickup"]}
                },
                "required": ["city", "method"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_return",
            "description": "Оформити заявку на повернення.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["order_id", "reason"],
                "additionalProperties": False
            }
        }
    }
]

def execute_tool(tool_name: str, args: dict, customer_id: str) -> str:
    try:
        if tool_name == "get_order":
            # 1. Запитуємо замовлення у внутрішнього сервісу (приймає лише order_id)
            order = service.get_order(args["order_id"])
            
            # 2. ПЕРЕВІРКА ВЛАСНИКА: чи належить замовлення поточному клієнту
            if order.get("customer_id") != customer_id:
                return json.dumps({"error": "Замовлення не знайдено або доступ заборонено", "code": "not_found"}, ensure_ascii=False)
            
            # 3. ФІЛЬТРАЦІЯ ВИТОКІВ: видаляємо службові примітки
            if "internal_note" in order:
                del order["internal_note"]
            return json.dumps(order, ensure_ascii=False)
            
        elif tool_name == "search_catalog":
            return json.dumps(service.search_catalog(args["query"]), ensure_ascii=False)
            
        elif tool_name == "check_stock":
            return json.dumps(service.check_stock(args["sku"]), ensure_ascii=False)
            
        elif tool_name == "calculate_delivery":
            # Передаємо items згідно з вашою JSON Schema
            return json.dumps(service.calculate_delivery(args["city"], args["method"], args.get("items", [])), ensure_ascii=False)
            
        elif tool_name == "request_return":
            # Перед поверненням товару перевіряємо, чи належить замовлення клієнту
            order = service.get_order(args["order_id"])
            if order.get("customer_id") != customer_id:
                return json.dumps({"error": "Замовлення не знайдено або доступ заборонено", "code": "not_found"}, ensure_ascii=False)
            
            # Виконуємо повернення з розширеними аргументами
            return json.dumps(service.request_return(args["order_id"], args["sku"], args["reason"], args.get("quantity", 1)), ensure_ascii=False)
            
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}", "code": "unknown_tool"}, ensure_ascii=False)
            
    except service.ShopError as e:
        return json.dumps({"error": str(e), "code": getattr(e, "code", "error")}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "code": "internal_error"}, ensure_ascii=False)