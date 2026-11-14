import sys
import json
import time
from pathlib import Path

# Додаємо корінь проєкту в sys.path для імпорту модуля app
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.inference import generate_dialog_response
from fastapi import HTTPException


def _is_provider_failure(error):
    return isinstance(error, HTTPException) and error.status_code in {
        401, 403, 429, 502, 503, 504
    }


def _save_report(report_data, output_path, eval_dir):
    out_file = Path(output_path) if output_path else eval_dir / "comparison_results.json"
    out_file.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_file

def run_from_requests_json(
    requests_path=None,
    output_path=None,
    request_fn=None,
    delay_seconds=1.05,
):
    eval_dir = Path(__file__).resolve().parent
    json_path = Path(requests_path) if requests_path else eval_dir / "requests.json"
    generate = request_fn or generate_dialog_response
    
    if not json_path.exists():
        print(f"[X] Не знайдено файл {json_path}. Перевірте наявність файлу з тестами.")
        return None

    data = json.loads(json_path.read_text(encoding="utf-8"))
    single_queries = data.get("звернення", [])
    dialogs = data.get("діалоги", [])

    modes = ["fuzzy_query", "structured_prompt", "context_engineering"]
    report_data = {"single_queries": {}, "dialogs": {}}
    abort_reason = None

    print("=== Запуск захищеного порівняння режимів ПР4 ===")

    for mode in modes:
        print(f"\n--- Режим: {mode} ---")
        report_data["single_queries"][mode] = []
        
        for idx, item in enumerate(single_queries, start=1):
            query_text = item.get("текст")
            print(f"  [Звернення {idx}] ({item.get('вид')}): {query_text[:35]}...")
            try:
                res = generate(history=[], user_query=query_text, mode=mode)
                report_data["single_queries"][mode].append({
                    "id": f"S{idx}",
                    "type": item.get("вид"),
                    "query": query_text,
                    "response_text": res.get("response_text", ""),
                    "schema_valid": res.get("schema_valid", False),
                    "parsed_data": res.get("parsed_data"),
                    "tokens": res.get("tokens", {}),
                    "time_sec": res.get("execution_time_seconds", 0.0),
                })
            except Exception as e:
                print(f"    [!] Помилка у зверненні {idx}: {e}")
                report_data["single_queries"][mode].append({
                    "id": f"S{idx}",
                    "type": item.get("вид"),
                    "query": query_text,
                    "error": str(e),
                })
                if _is_provider_failure(e):
                    abort_reason = str(e.detail)
                    break
            if delay_seconds:
                if abort_reason is None:
                    time.sleep(delay_seconds) # Пауза проти Rate Limit (429)

        if abort_reason is not None:
            break

        report_data["dialogs"][mode] = []
        for idx, dialog in enumerate(dialogs, start=1):
            turns = dialog.get("репліки", [])
            print(f"  [Діалог {idx}] ({dialog.get('вид')} - {len(turns)} реплік)...")
            history = []
            turns_log = []
            
            for turn_idx, turn_text in enumerate(turns, start=1):
                try:
                    res = generate(history=history, user_query=turn_text, mode=mode)
                    if mode == "context_engineering":
                        history.append({"role": "user", "content": turn_text})
                        history.append({"role": "assistant", "content": res["response_text"]})
                    
                    turns_log.append({
                        "turn": turn_idx,
                        "user": turn_text,
                        "response_text": res.get("response_text", ""),
                        "schema_valid": res.get("schema_valid", False),
                        "parsed_data": res.get("parsed_data"),
                        "tokens": res.get("tokens", {}),
                        "time_sec": res.get("execution_time_seconds", 0.0),
                    })
                except Exception as e:
                    print(f"    [!] Помилка у репліці {turn_idx}: {e}")
                    turns_log.append({
                        "turn": turn_idx,
                        "user": turn_text,
                        "error": str(e),
                    })
                    if _is_provider_failure(e):
                        abort_reason = str(e.detail)
                        break
                if delay_seconds:
                    if abort_reason is None:
                        time.sleep(delay_seconds)

            report_data["dialogs"][mode].append({
                "id": f"D{idx}",
                "description": dialog.get("вид"),
                "checks": dialog.get("перевіряє"),
                "turns": turns_log,
            })
            if abort_reason is not None:
                break

        if abort_reason is not None:
            break

    if abort_reason is not None:
        report_data["aborted"] = {"reason": abort_reason}
    out_file = _save_report(report_data, output_path, eval_dir)
    if abort_reason is not None:
        print(f"\n[X] Порівняння зупинено: {abort_reason}")
        print(f"[i] Частковий звіт збережено у {out_file}")
    else:
        print(f"\n[OK] Результати порівняння успішно збережено у {out_file}")
    return report_data

if __name__ == "__main__":
    result = run_from_requests_json()
    if result and result.get("aborted"):
        sys.exit(1)