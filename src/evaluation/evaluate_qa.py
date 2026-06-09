from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List

from src.evaluation.eval_logger import EvalLogger
from src.qa_pipeline import LegalQAPipeline

LOGGER = logging.getLogger(__name__)


def _coerce_int_like(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    return None
    return None


def _load_question_rows_from_jsonl(text: str, *, source: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} in {source}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number} in {source} must be a JSON object.")

        question = str(row.get("question") or "").strip()
        if not question:
            question_id = row.get("id", row.get("question_id", line_number))
            raise ValueError(f"Question row {question_id} in {source} is missing question text.")
        rows.append(row)
    return rows


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
                rows: List[Dict[str, object]] = []
                for index, row in enumerate(parsed, start=1):
                    if not isinstance(row, dict):
                        raise ValueError(f"Item {index} in {question_path} must be a JSON object.")
                    question = str(row.get("question") or "").strip()
                    if not question:
                        question_id = row.get("id", row.get("question_id", index))
                        raise ValueError(f"Question row {question_id} in {question_path} is missing question text.")
                    rows.append(row)
                return rows
            if isinstance(parsed, dict):
                question = str(parsed.get("question") or "").strip()
                if not question:
                    question_id = parsed.get("id", parsed.get("question_id", 1))
                    raise ValueError(f"Question row {question_id} in {question_path} is missing question text.")
                return [parsed]

    return _load_question_rows_from_jsonl(raw_text, source=str(question_path))


def _slice_questions(questions: List[Dict[str, object]], limit: int | None) -> List[Dict[str, object]]:
    if limit is None:
        return questions
    if limit <= 0:
        return []
    return questions[:limit]


def _normalize_submission_id(row: Dict[str, object], fallback_id: int) -> int:
    for key in ("id", "question_id", "sample_id", "qid"):
        normalized = _coerce_int_like(row.get(key))
        if normalized is not None:
            return normalized
    return fallback_id


def _build_relevant_docs(result: Dict[str, object]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for item in list(result.get("relevant_doc_details") or []):
        if not isinstance(item, dict):
            continue
        doc_title = str(item.get("doc_title") or "").strip()
        source_url = str(item.get("source_url") or "").strip()
        citation = str(item.get("citation") or doc_title).strip()
        key = (doc_title, source_url)
        if key in seen:
            continue
        if not doc_title and not source_url and not citation:
            continue
        seen.add(key)
        records.append(
            {
                "doc_title": doc_title,
                "source_url": source_url,
                "citation": citation,
            }
        )

    if records:
        return records

    for item in list(result.get("citations") or []):
        if not isinstance(item, dict):
            continue
        doc_title = str(item.get("doc_title") or "").strip()
        source_url = str(item.get("source_url") or "").strip()
        citation = str(item.get("citation") or doc_title).strip()
        key = (doc_title, source_url)
        if key in seen:
            continue
        if not doc_title and not source_url and not citation:
            continue
        seen.add(key)
        records.append(
            {
                "doc_title": doc_title,
                "source_url": source_url,
                "citation": citation,
            }
        )

    if records:
        return records

    for context in list(result.get("final_contexts") or []):
        if not isinstance(context, dict):
            continue
        metadata = dict(context.get("metadata") or {})
        doc_title = str(metadata.get("doc_title") or metadata.get("doc_id") or "").strip()
        source_url = str(metadata.get("source_url") or "").strip()
        citation = str(metadata.get("citation") or doc_title).strip()
        key = (doc_title, source_url)
        if key in seen:
            continue
        if not doc_title and not source_url and not citation:
            continue
        seen.add(key)
        records.append(
            {
                "doc_title": doc_title,
                "source_url": source_url,
                "citation": citation,
            }
        )

    return records


def _build_relevant_articles(result: Dict[str, object]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for item in list(result.get("relevant_article_details") or []):
        if not isinstance(item, dict):
            continue
        doc_title = str(item.get("doc_title") or "").strip()
        article = str(item.get("article") or "").strip()
        clause_value = item.get("clause")
        clause = "" if clause_value in (None, "") else str(clause_value).strip()
        citation = str(item.get("citation") or doc_title).strip()
        source_url = str(item.get("source_url") or "").strip()
        key = (doc_title, article, clause, citation, source_url)
        if key in seen:
            continue
        if not doc_title and not article and not clause and not citation and not source_url:
            continue
        seen.add(key)
        records.append(
            {
                "doc_title": doc_title,
                "article": article,
                "clause": clause if clause else None,
                "citation": citation,
                "source_url": source_url,
            }
        )

    if records:
        return records

    for item in list(result.get("citations") or []):
        if not isinstance(item, dict):
            continue
        doc_title = str(item.get("doc_title") or "").strip()
        article = str(item.get("article") or "").strip()
        clause_value = item.get("clause")
        clause = "" if clause_value in (None, "") else str(clause_value).strip()
        citation = str(item.get("citation") or doc_title).strip()
        source_url = str(item.get("source_url") or "").strip()
        key = (doc_title, article, clause, citation, source_url)
        if key in seen:
            continue
        if not doc_title and not article and not clause and not citation and not source_url:
            continue
        seen.add(key)
        records.append(
            {
                "doc_title": doc_title,
                "article": article,
                "clause": clause if clause else None,
                "citation": citation,
                "source_url": source_url,
            }
        )

    if records:
        return records

    for context in list(result.get("final_contexts") or []):
        if not isinstance(context, dict):
            continue
        metadata = dict(context.get("metadata") or {})
        doc_title = str(metadata.get("doc_title") or metadata.get("doc_id") or "").strip()
        article = str(metadata.get("article") or "").strip()
        clause_value = metadata.get("clause")
        clause = "" if clause_value in (None, "") else str(clause_value).strip()
        citation = str(metadata.get("citation") or doc_title).strip()
        source_url = str(metadata.get("source_url") or "").strip()
        key = (doc_title, article, clause, citation, source_url)
        if key in seen:
            continue
        if not doc_title and not article and not clause and not citation and not source_url:
            continue
        seen.add(key)
        records.append(
            {
                "doc_title": doc_title,
                "article": article,
                "clause": clause if clause else None,
                "citation": citation,
                "source_url": source_url,
            }
        )

    return records


def _evaluate_row(qa: LegalQAPipeline, row: Dict[str, object], index: int) -> Dict[str, object]:
    question_id = _normalize_submission_id(row, index)
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError(f"Question row {question_id} is missing question text.")

    started = time.perf_counter()
    result = qa.answer(question)
    latency = time.perf_counter() - started

    return {
        "id": question_id,
        "question": question,
        "result": result,
        "latency_seconds": latency,
        "row": row,
    }


def _evaluate_row_payload(qa: LegalQAPipeline, payload: tuple[int, Dict[str, object]]) -> Dict[str, object]:
    index, row = payload
    return _evaluate_row(qa, row, index)


def evaluate_questions(
    questions: List[Dict[str, object]],
    *,
    run_id: str,
    output_path: str | Path | None = None,
    answers_output_path: str | Path | None = None,
    limit: int | None = None,
) -> Dict[str, object]:
    export_path = output_path if output_path is not None else answers_output_path
    logger = EvalLogger(run_id=run_id)
    qa = LegalQAPipeline()
    route_distribution: Counter[str] = Counter()
    answer_non_empty = 0
    citation_present = 0
    total_contexts = 0
    latencies: List[float] = []
    legal_ref_hits = 0
    legal_ref_total = 0
    answer_records: List[Dict[str, object]] = []

    selected_questions = _slice_questions(questions, limit)
    worker_count = max(1, min(int(os.getenv("R2AI_EVAL_WORKERS", "1")), len(selected_questions) or 1))
    if worker_count == 1:
        evaluated_rows = [_evaluate_row(qa, row, index) for index, row in enumerate(selected_questions, start=1)]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            evaluated_rows = list(executor.map(partial(_evaluate_row_payload, qa), enumerate(selected_questions, start=1)))

    for item in evaluated_rows:
        row = item["row"]
        result = item["result"]
        question_id = item["id"]
        question = item["question"]
        latency = item["latency_seconds"]
        answer_text = str(result.get("answer") or "")

        latencies.append(latency)
        route_distribution[str(result["route"])] += 1
        total_contexts += len(result.get("final_contexts") or [])
        if answer_text.strip():
            answer_non_empty += 1
        if result.get("citations"):
            citation_present += 1

        expected_refs = list(row.get("expected_law_refs") or [])
        if expected_refs:
            legal_ref_total += 1
            if any(ref in answer_text for ref in expected_refs):
                legal_ref_hits += 1

        if not answer_text.strip():
            LOGGER.warning("Empty answer generated for question_id=%s", question_id)

        relevant_docs = _build_relevant_docs(result)
        relevant_articles = _build_relevant_articles(result)
        if not relevant_docs and not relevant_articles and (
            result.get("citations") or result.get("final_contexts")
        ):
            LOGGER.warning("Could not derive structured references for question_id=%s", question_id)

        answer_records.append(
            {
                "id": question_id,
                "question": question,
                "answer": answer_text,
                "relevant_docs": relevant_docs,
                "relevant_articles": relevant_articles,
            }
        )

        logger.log(
            {
                "question_id": row.get("question_id", row.get("id")),
                "question": question,
                "route": result.get("route"),
                "domains": result.get("domains"),
                "seed_chunk_ids": [item.get("chunk_id") for item in result.get("retrieved_chunks", [])],
                "expanded_context_ids": [item.get("chunk_id") for item in result.get("expanded_contexts", [])],
                "final_context_ids": [item.get("chunk_id") for item in result.get("final_contexts", [])],
                "final_contexts": result.get("final_contexts"),
                "citations": result.get("citations"),
                "relevant_docs": result.get("relevant_doc_details") or result.get("relevant_docs"),
                "relevant_articles": result.get("relevant_article_details") or result.get("relevant_articles"),
                "answer": answer_text,
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

    if export_path is not None:
        output_file = Path(export_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(answer_records, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the legal QA evaluation/export pipeline.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of questions to run.")
    args = parser.parse_args()
    summary = evaluate_questions(
        load_questions(args.questions),
        run_id=args.run_id,
        output_path=args.output,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
