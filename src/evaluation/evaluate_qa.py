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
    question_path = Path(path)
    if not question_path.exists():
        return []

    raw_text = question_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    if raw_text[0] in "[{":
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = None
        else:
            if isinstance(parsed, list):
                return [row for row in parsed if isinstance(row, dict)]
            if isinstance(parsed, dict):
                return [parsed]

    rows: List[Dict[str, object]] = []
    with question_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _slice_questions(questions: List[Dict[str, object]], limit: int | None) -> List[Dict[str, object]]:
    if limit is None:
        return questions
    if limit <= 0:
        return []
    return questions[:limit]


def evaluate_questions(
    questions: List[Dict[str, object]],
    *,
    run_id: str,
    answers_output_path: str | Path | None = None,
    limit: int | None = None,
) -> Dict[str, object]:
    qa = LegalQAPipeline()
    logger = EvalLogger(run_id=run_id)
    route_distribution: Counter[str] = Counter()
    answer_non_empty = 0
    citation_present = 0
    total_contexts = 0
    latencies: List[float] = []
    legal_ref_hits = 0
    legal_ref_total = 0
    answer_records: List[Dict[str, object]] = []

    selected_questions = _slice_questions(questions, limit)
    for row in selected_questions:
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

        record = {
            "question_id": row.get("question_id", row.get("id")),
            "question": row.get("question"),
            "route": result.get("route"),
            "domains": result.get("domains"),
            "answer": result.get("answer"),
            "citations": result.get("citations"),
            "relevant_docs": result.get("relevant_docs"),
            "relevant_articles": result.get("relevant_articles"),
            "seed_contexts": result.get("seed_contexts"),
            "expanded_contexts": result.get("expanded_contexts"),
            "final_contexts": result.get("final_contexts"),
            "grounding": result.get("grounding"),
            "seed_chunk_ids": [item.get("chunk_id") for item in result.get("retrieved_chunks", [])],
            "expanded_context_ids": [item.get("chunk_id") for item in result.get("expanded_contexts", [])],
            "final_context_ids": [item.get("chunk_id") for item in result.get("final_contexts", [])],
            "latency_seconds": latency,
        }
        answer_records.append(record)

        logger.log(
            {
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "route": result.get("route"),
                "domains": result.get("domains"),
                "seed_chunk_ids": [item.get("chunk_id") for item in result.get("retrieved_chunks", [])],
                "expanded_context_ids": [item.get("chunk_id") for item in result.get("expanded_contexts", [])],
                "final_context_ids": [item.get("chunk_id") for item in result.get("final_contexts", [])],
                "final_contexts": result.get("final_contexts"),
                "citations": result.get("citations"),
                "relevant_docs": result.get("relevant_docs"),
                "relevant_articles": result.get("relevant_articles"),
                "answer": result.get("answer"),
                "grounding": result.get("grounding"),
                "latency_seconds": latency,
            }
        )

    total_questions = len(selected_questions)
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

    if answers_output_path is not None:
        output_path = Path(answers_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "summary": summary,
            "answers": answer_records,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run smoke evaluation for the legal QA pipeline.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--answers-output", default=None, help="Optional JSON file for the full answer dump.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of questions to run.")
    args = parser.parse_args()
    summary = evaluate_questions(
        load_questions(args.questions),
        run_id=args.run_id,
        answers_output_path=args.answers_output,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
