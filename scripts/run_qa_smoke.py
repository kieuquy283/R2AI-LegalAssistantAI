from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qa_pipeline import LegalQAPipeline


QUERIES = [
    "Ai khong duoc thanh lap doanh nghiep?",
    "Khong gop du von dieu le dung han thi bi phat gi?",
    "Nguoi nuoc ngoai gop von vao cong ty Viet Nam can dieu kien gi?",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    qa = LegalQAPipeline()
    results = []
    for query in QUERIES:
        result = qa.answer(query)
        results.append(
            {
                "question": query,
                "route": result["route"],
                "citations": result["citations"],
                "answer": result["answer"],
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
