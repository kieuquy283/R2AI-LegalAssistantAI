from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate_qa import evaluate_questions, load_questions


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    run_id = f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    questions = load_questions("data/evaluation/sample_questions.jsonl")
    summary = evaluate_questions(questions, run_id=run_id)
    print(json.dumps({"run_id": run_id, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
