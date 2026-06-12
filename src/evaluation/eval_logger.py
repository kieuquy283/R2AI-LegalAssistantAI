from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


class EvalLogger:
    def __init__(self, run_id: str, *, logs_dir: str | Path = "logs/eval_runs") -> None:
        self.run_id = run_id
        self.logs_dir = Path(logs_dir)
        self.path = self.logs_dir / f"{run_id}.jsonl"
        self.progress_path = self.logs_dir / f"{run_id}_progress.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Dict[str, object]) -> None:
        record = dict(payload)
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_progress(self, message: str, **payload: object) -> None:
        record = {"event": "progress", "message": message}
        record.update(payload)
        self.log(record)
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")
