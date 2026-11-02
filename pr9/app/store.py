import json
from pathlib import Path

RUNS_DIR = Path(__file__).parent.parent / "runs"
RUNS_DIR.mkdir(exist_ok=True, parents=True)

def _get_run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"

def save_state(state: dict) -> None:
    run_id = state.get("run_id")
    if not run_id:
        return
    path = _get_run_path(run_id)
    temp_path = path.with_suffix(".tmp")
    # Атомарний запис через тимчасовий файл
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)

def load_state(run_id: str) -> dict | None:
    path = _get_run_path(run_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def list_runs() -> list[dict]:
    runs = []
    for path in RUNS_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                runs.data.append(data) if hasattr(runs, "data") else runs.append(data)
        except Exception:
            continue
    # Сортировка за новизною або ID
    return sorted(runs, key=lambda x: x.get("created_at", ""), reverse=True)