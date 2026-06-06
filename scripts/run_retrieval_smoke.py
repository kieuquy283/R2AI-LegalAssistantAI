from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.answer_generator import AnswerGenerator
from src.retrieval.retrieval_pipeline import RetrievalPipeline


QUERY = "Ai khong duoc thanh lap doanh nghiep?"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    pipeline = RetrievalPipeline()
    retrieval = pipeline.run(QUERY)
    answer = AnswerGenerator().generate(query=QUERY, retrieval_result=retrieval)
    payload = {
        "query": QUERY,
        "route": retrieval["route"],
        "top_chunks": retrieval["seed_chunks"][:3],
        "final_contexts": retrieval["final_contexts"][:5],
        "citations": answer["citations"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
