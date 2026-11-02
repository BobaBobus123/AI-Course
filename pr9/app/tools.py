import json
from shop import service

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Отримати інформацію про замовлення за його номером.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Номер замовлення"}
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
            "description": "Знайти товар у каталозі магазину.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Назва товару"}
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
            "description": "Перевірити наявність товару на складі за його артикулом.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Артикул товару"}
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
            "description": "Розрахувати вартість та строк доставки.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Місто доставки"},
                    "method": {"type": "string", "enum": ["branch", "courier", "pickup"], "description": "Спосіб доставки"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "quantity": {"type": "integer"}
                            },
                            "required": ["sku", "quantity"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["city", "method", "items"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_return",
            "description": "Оформити заявку на повернення товару.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "sku": {"type": "string"},
                    "reason": {"type": "string", "enum": ["not_suitable", "defect", "wrong_item", "damaged"]},
                    "quantity": {"type": "integer"}
                },
                "required": ["order_id", "sku", "reason"],
                "additionalProperties": False
            }
        }
    }
]

def execute_tool(tool_name: str, args: dict, customer_id: str) -> str:
    try:
        if tool_name == "get_order":
            order = service.get_order(args["order_id"])
            if order.get("customer_id") != customer_id:
                return json.dumps({"error": "Замовлення не знайдено або доступ заборонено", "code": "not_found"}, ensure_ascii=False)
            if "internal_note" in order:
                del order["internal_note"]
            return json.dumps(order, ensure_ascii=False)
            
        elif tool_name == "search_catalog":
            return json.dumps(service.search_catalog(args["query"]), ensure_ascii=False)
            
        elif tool_name == "check_stock":
            return json.dumps(service.check_stock(args["sku"]), ensure_ascii=False)
            
        elif tool_name == "calculate_delivery":
            return json.dumps(service.calculate_delivery(args["city"], args["method"], args.get("items", [])), ensure_ascii=False)
            
        elif tool_name == "request_return":
            order = service.get_order(args["order_id"])
            if order.get("customer_id") != customer_id:
                return json.dumps({"error": "Замовлення не знайдено або доступ заборонено", "code": "not_found"}, ensure_ascii=False)
            return json.dumps(service.request_return(args["order_id"], args["sku"], args["reason"], args.get("quantity", 1)), ensure_ascii=False)
            
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}", "code": "unknown_tool"}, ensure_ascii=False)
            
    except service.ShopError as e:
        return json.dumps({"error": str(e), "code": getattr(e, "code", "error")}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "code": "internal_error"}, ensure_ascii=False)