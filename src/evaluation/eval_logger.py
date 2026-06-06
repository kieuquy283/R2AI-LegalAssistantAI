from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


class EvalLogger:
    def __init__(self, run_id: str, *, logs_dir: str | Path = "logs/eval_runs") -> None:
        self.run_id = run_id
        self.path = Path(logs_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Dict[str, object]) -> None:
        record = dict(payload)
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
