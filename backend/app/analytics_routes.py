"""Read-only API exposing saved DarijaFlow experiment artifacts to the frontend."""
from pathlib import Path
import csv, json
from typing import Any
from fastapi import APIRouter

router = APIRouter(prefix="/analytics", tags=["analytics"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results_experiment"
LABELS = {"vanilla": "Vanilla LSTM", "cifg": "CIFG LSTM", "wmc": "WMC LSTM"}

def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows=[]
        for row in csv.DictReader(f):
            clean={}
            for k,v in row.items():
                if v is None or v == "": clean[k]=None
                else:
                    try: clean[k]=float(v) if any(ch in v for ch in ".eE") else int(v)
                    except (ValueError, TypeError): clean[k]=v
            rows.append(clean)
        return rows

def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default
    except (json.JSONDecodeError, OSError):
        return default

@router.get("/data")
def analytics_data():
    summary = read_json(RESULTS / "metrics" / "training_summary.json", [])
    test_rows = read_json(RESULTS / "metrics" / "test_comparison.json", [])
    if not test_rows:
        test_rows = read_csv(RESULTS / "metrics" / "test_comparison.csv")
    test_by_model = {str(r.get("model", r.get("variant", ""))).lower(): r for r in test_rows if isinstance(r, dict)}
    summary_by_model = {str(r.get("variant", r.get("model", ""))).lower(): r for r in summary if isinstance(r, dict)}
    models=[]
    sources=[]
    for key in ("vanilla", "cifg", "wmc"):
        history_path = RESULTS / "history" / f"{key}_history.csv"
        history = read_csv(history_path)
        if history_path.exists(): sources.append(str(history_path.relative_to(PROJECT_ROOT)))
        test = test_by_model.get(key, {})
        summ = summary_by_model.get(key, {})
        models.append({"model": key, "history": history, "test_metrics": test, "metrics": summ, "summary": summ})
    examples_path = RESULTS / "examples" / "test_translation_examples.csv"
    examples = read_csv(examples_path)
    if examples_path.exists(): sources.append(str(examples_path.relative_to(PROJECT_ROOT)))
    for p in (RESULTS / "metrics" / "test_comparison.json", RESULTS / "metrics" / "test_comparison.csv", RESULTS / "metrics" / "training_summary.json"):
        if p.exists(): sources.append(str(p.relative_to(PROJECT_ROOT)))
    return {"models": models, "examples": examples, "sources": sources}
