from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

from src.evaluation.eval_logger import EvalLogger
from src.qa_pipeline import LegalQAPipeline


def load_questions(path: str | Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    question_path = Path(path)
    if not question_path.exists():
        return rows
    with question_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def evaluate_questions(questions: List[Dict[str, object]], *, run_id: str) -> Dict[str, object]:
    qa = LegalQAPipeline()
    logger = EvalLogger(run_id=run_id)
    route_distribution: Counter[str] = Counter()
    answer_non_empty = 0
    citation_present = 0
    total_contexts = 0
    latencies: List[float] = []
    legal_ref_hits = 0
    legal_ref_total = 0

    for row in questions:
        started = time.perf_counter()
        result = qa.answer(str(row["question"]))
        latency = time.perf_counter() - started
        latencies.append(latency)
        route_distribution[str(result["route"])] += 1
        total_contexts += len(result.get("final_contexts") or [])
        if result.get("answer"):
            answer_non_empty += 1
        if result.get("citations"):
            citation_present += 1

        expected_refs = list(row.get("expected_law_refs") or [])
        if expected_refs:
            legal_ref_total += 1
            answer_text = str(result.get("answer") or "")
            if any(ref in answer_text for ref in expected_refs):
                legal_ref_hits += 1

        logger.log(
            {
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "route": result.get("route"),
                "domains": result.get("domains"),
                "seed_chunk_ids": [item.get("chunk_id") for item in result.get("retrieved_chunks", [])],
                "expanded_context_ids": [item.get("chunk_id") for item in result.get("expanded_contexts", [])],
                "final_context_ids": [item.get("chunk_id") for item in result.get("final_contexts", [])],
                "citations": result.get("citations"),
                "answer": result.get("answer"),
                "grounding": result.get("grounding"),
                "latency_seconds": latency,
            }
        )

    total_questions = len(questions)
    summary = {
        "total_questions": total_questions,
        "citation_present_rate": citation_present / total_questions if total_questions else 0.0,
        "answer_non_empty_rate": answer_non_empty / total_questions if total_questions else 0.0,
        "route_distribution": dict(route_distribution),
        "avg_context_count": total_contexts / total_questions if total_questions else 0.0,
        "avg_latency_seconds": sum(latencies) / total_questions if total_questions else 0.0,
        "legal_ref_hit_rate": legal_ref_hits / legal_ref_total if legal_ref_total else None,
    }
    summary_path = Path("logs/eval_runs") / f"{run_id}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run smoke evaluation for the legal QA pipeline.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    summary = evaluate_questions(load_questions(args.questions), run_id=args.run_id)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
