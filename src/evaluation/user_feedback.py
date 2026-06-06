from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from uuid import uuid4


class UserFeedbackStore:
    def __init__(self, *, path: str | Path = "logs/user_feedback/feedback.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_feedback(
        self,
        *,
        question: str,
        answer: str,
        rating: str,
        comment: str | None = None,
    ) -> Dict[str, object]:
        record = {
            "feedback_id": uuid4().hex,
            "question": question,
            "answer": answer,
            "rating": rating,
            "comment": comment or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def list_feedback(self) -> List[Dict[str, object]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, object]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Append one user feedback record.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--rating", required=True)
    parser.add_argument("--comment", default="")
    args = parser.parse_args()
    record = UserFeedbackStore().add_feedback(
        question=args.question,
        answer=args.answer,
        rating=args.rating,
        comment=args.comment,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
