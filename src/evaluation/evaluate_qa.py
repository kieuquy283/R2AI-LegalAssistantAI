from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List

from src.evaluation.output_formatter import extract_legal_doc_code, format_article_ref, format_doc_ref, format_submission_record
from src.evaluation.eval_logger import EvalLogger
from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator, evaluate_batch
from src.qa_pipeline import LegalQAPipeline

LOGGER = logging.getLogger(__name__)
LEGAL_CODE_PATTERN = re.compile(r"\b\d+(?:/\d+)+/[A-Z0-9ĂÂĐÊÔƠƯ\-]+(?:[A-Z0-9ĂÂĐÊÔƠƯ\-]+)?\b", re.IGNORECASE)


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


def _extract_legal_doc_code(*values: object) -> str:
    return extract_legal_doc_code(*values)


def _format_doc_reference(doc_id: str, doc_title: str, citation: str = "") -> str:
    return format_doc_ref({"doc_id": doc_id, "doc_title": doc_title, "citation": citation})


def _format_article_reference(doc_id: str, doc_title: str, article: str, citation: str = "") -> str:
    return format_article_ref({"doc_id": doc_id, "doc_title": doc_title, "article": article, "citation": citation})


def _context_metadata(context: Dict[str, object]) -> Dict[str, object]:
    merged = dict(context)
    nested = dict(context.get("metadata") or {})
    merged.update({key: value for key, value in nested.items() if value not in (None, "", [])})
    return merged


def _build_relevant_docs(result: Dict[str, object]) -> List[str]:
    records: List[str] = []
    seen: set[str] = set()

    for item in list(result.get("relevant_doc_details") or []):
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        doc_title = str(item.get("doc_title") or "").strip()
        citation = str(item.get("citation") or "").strip()
        ref = _format_doc_reference(doc_id, doc_title, citation)
        if not ref or ref in seen:
            continue
        seen.add(ref)
        records.append(ref)

    if records:
        return records

    for item in list(result.get("citations") or []):
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        doc_title = str(item.get("doc_title") or "").strip()
        citation = str(item.get("citation") or "").strip()
        ref = _format_doc_reference(doc_id, doc_title, citation)
        if not ref or ref in seen:
            continue
        seen.add(ref)
        records.append(ref)

    if records:
        return records

    for context in list(result.get("final_contexts") or []):
        if not isinstance(context, dict):
            continue
        metadata = _context_metadata(context)
        doc_id = str(metadata.get("doc_id") or "").strip()
        doc_title = str(metadata.get("doc_title") or metadata.get("doc_id") or "").strip()
        citation = str(metadata.get("citation") or "").strip()
        ref = _format_doc_reference(doc_id, doc_title, citation)
        if not ref or ref in seen:
            continue
        seen.add(ref)
        records.append(ref)

    return records


def _build_relevant_articles(result: Dict[str, object]) -> List[str]:
    records: List[str] = []
    seen: set[str] = set()

    for item in list(result.get("relevant_article_details") or []):
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        doc_title = str(item.get("doc_title") or "").strip()
        article = str(item.get("article") or "").strip()
        citation = str(item.get("citation") or "").strip()
        ref = _format_article_reference(doc_id, doc_title, article, citation)
        if not ref or ref in seen:
            continue
        seen.add(ref)
        records.append(ref)

    if records:
        return records

    for item in list(result.get("citations") or []):
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        doc_title = str(item.get("doc_title") or "").strip()
        article = str(item.get("article") or "").strip()
        citation = str(item.get("citation") or "").strip()
        ref = _format_article_reference(doc_id, doc_title, article, citation)
        if not ref or ref in seen:
            continue
        seen.add(ref)
        records.append(ref)

    if records:
        return records

    for context in list(result.get("final_contexts") or []):
        if not isinstance(context, dict):
            continue
        metadata = _context_metadata(context)
        doc_id = str(metadata.get("doc_id") or "").strip()
        doc_title = str(metadata.get("doc_title") or metadata.get("doc_id") or "").strip()
        article = str(metadata.get("article") or "").strip()
        citation = str(metadata.get("citation") or "").strip()
        ref = _format_article_reference(doc_id, doc_title, article, citation)
        if not ref or ref in seen:
            continue
        seen.add(ref)
        records.append(ref)

    return records


def _evaluate_row(qa: LegalQAPipeline, row: Dict[str, object], index: int) -> Dict[str, object]:
    question_id = _normalize_submission_id(row, index)
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError(f"Question row {question_id} is missing question text.")

    print(f"[{index}] Q{question_id}: {question[:80]}...")
    sys.stdout.flush()
    started = time.perf_counter()
    result = qa.answer(question)
    latency = time.perf_counter() - started
    contexts = len(result.get("final_contexts") or [])
    route = result.get("route", "?")
    print(f"[{index}] Done in {latency:.2f}s | route={route} | contexts={contexts} | answer={len(str(result.get('answer') or ''))} chars")
    sys.stdout.flush()

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


def _progress_interval(total_questions: int) -> int:
    configured = os.getenv("R2AI_EVAL_PROGRESS_EVERY", "").strip()
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    if total_questions >= 1000:
        return 25
    if total_questions >= 200:
        return 10
    return 5


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
    trace_path_value = os.getenv("R2AI_RETRIEVAL_TRACE_PATH", "").strip()
    trace_path = Path(trace_path_value) if trace_path_value else None
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")
    route_distribution: Counter[str] = Counter()
    answer_non_empty = 0
    citation_present = 0
    total_contexts = 0
    latencies: List[float] = []
    legal_ref_hits = 0
    legal_ref_total = 0
    answer_records: List[Dict[str, object]] = []

    selected_questions = _slice_questions(questions, limit)
    total_questions = len(selected_questions)
    progress_every = _progress_interval(total_questions)
    run_started = time.perf_counter()
    worker_count = max(1, min(int(os.getenv("R2AI_EVAL_WORKERS", "1")), len(selected_questions) or 1))
    print(f"\n{'='*60}")
    print(f"EVAL START: run_id={run_id} total_questions={total_questions}")
    print(f"output={export_path or ''} workers={worker_count}")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    logger.log_progress(
        f"START run_id={run_id} total_questions={total_questions} output={export_path or ''} workers={worker_count}",
        run_id=run_id,
        total_questions=total_questions,
        output_path=str(export_path or ""),
        workers=worker_count,
        progress_every=progress_every,
    )
    if worker_count == 1:
        evaluated_rows = [_evaluate_row(qa, row, index) for index, row in enumerate(selected_questions, start=1)]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            evaluated_rows = list(executor.map(partial(_evaluate_row_payload, qa), enumerate(selected_questions, start=1)))

    for processed_count, item in enumerate(evaluated_rows, start=1):
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
            format_submission_record(
                {
                    "id": question_id,
                    "question": question,
                    "answer": answer_text,
                    "relevant_docs": relevant_docs,
                    "relevant_articles": relevant_articles,
                }
            )
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
        if trace_path is not None:
            top_candidates = []
            for context in list(result.get("raw_final_contexts") or result.get("expanded_contexts") or [])[:10]:
                metadata = _context_metadata(context)
                top_candidates.append(
                    {
                        "retrieval_level": context.get("retrieval_level"),
                        "retrieval_source": context.get("retrieval_source", ""),
                        "doc_number": metadata.get("doc_number"),
                        "doc_title": metadata.get("doc_title"),
                        "article": metadata.get("article"),
                        "citation": metadata.get("citation"),
                        "domain": metadata.get("domain"),
                        "raw_dense_score": context.get("raw_dense_score", 0.0),
                        "dense_score": context.get("dense_score", 0.0),
                        "score": context.get("score", context.get("raw_dense_score", 0.0)),
                        "bm25_score": context.get("bm25_score", 0.0),
                        "title_overlap": context.get("title_overlap", 0.0),
                        "lexical_overlap": context.get("lexical_overlap", 0.0),
                        "final_score": context.get("final_score", context.get("score", 0.0)),
                        "confidence": context.get("confidence", 0.0),
                        "qdrant_mode": context.get("qdrant_mode", ""),
                        "domain_rerank_enabled": context.get("domain_rerank_enabled", False),
                        "domain_rerank_mode": context.get("domain_rerank_mode", ""),
                        "detected_query_domain": context.get("detected_query_domain"),
                        "detected_domains": context.get("detected_domains"),
                        "primary_domain": context.get("primary_domain"),
                        "is_multi_domain": context.get("is_multi_domain", False),
                        "domain_confidence": context.get("domain_confidence"),
                        "matched_domain_keywords": context.get("matched_domain_keywords"),
                        "candidate_domain": context.get("candidate_domain"),
                        "score_before_domain_adjustment": context.get("score_before_domain_adjustment", 0.0),
                        "score_after_domain_adjustment": context.get("score_after_domain_adjustment", 0.0),
                        "domain_adjustment_reason": context.get("domain_adjustment_reason", ""),
                    }
                )
            trace_row = {
                "question_id": question_id,
                "question": question,
                "route": result.get("route"),
                "domain_prediction": {"domains": result.get("domains")},
                "candidate_count": len(result.get("expanded_contexts") or []),
                "selected_context_count": len(result.get("final_contexts") or []),
                "selected_docs": relevant_docs,
                "selected_articles": relevant_articles,
                "qdrant_path": os.getenv("QDRANT_PATH", ""),
                "qdrant_mode": top_candidates[0].get("qdrant_mode", "") if top_candidates else "",
                "top_candidates": top_candidates,
            }
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace_row, ensure_ascii=False) + "\n")

        if processed_count % progress_every == 0 or processed_count == total_questions:
            elapsed = time.perf_counter() - run_started
            avg_latency = elapsed / processed_count if processed_count else 0.0
            remaining = max(total_questions - processed_count, 0)
            eta_seconds = remaining * avg_latency
            message = (
                f"[PROGRESS] {processed_count}/{total_questions} "
                f"({processed_count / total_questions:.1%}) "
                f"avg={avg_latency:.2f}s/question eta={eta_seconds / 60:.1f}m"
            )
            print(f"\n{'-'*60}")
            print(message)
            print(f"{'-'*60}\n")
            sys.stdout.flush()
            logger.log_progress(
                message,
                run_id=run_id,
                processed=processed_count,
                total_questions=total_questions,
                percent_complete=round(processed_count / total_questions, 4),
                avg_latency_seconds=round(avg_latency, 4),
                eta_seconds=round(eta_seconds, 2),
                last_question_id=question_id,
            )

    # Comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    comprehensive_metrics = evaluate_batch(
        evaluated_rows,
        gold_data=selected_questions,
        auto_eval=False,
    )
    
    summary = {
        "total_questions": total_questions,
        "citation_present_rate": citation_present / total_questions if total_questions else 0.0,
        "answer_non_empty_rate": answer_non_empty / total_questions if total_questions else 0.0,
        "route_distribution": dict(route_distribution),
        "avg_context_count": total_contexts / total_questions if total_questions else 0.0,
        "avg_latency_seconds": sum(latencies) / total_questions if total_questions else 0.0,
        "legal_ref_hit_rate": legal_ref_hits / legal_ref_total if legal_ref_total else None,
        "comprehensive_evaluation": comprehensive_metrics,
    }
    summary_path = Path("logs/eval_runs") / f"{run_id}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    total_elapsed = time.perf_counter() - run_started
    print(f"\n{'='*60}")
    print(f"EVAL DONE: run_id={run_id}")
    print(f"total_questions={total_questions} elapsed={total_elapsed / 60:.1f}m")
    print(f"avg_latency={summary['avg_latency_seconds']:.2f}s")
    print(f"\nCOMPREHENSIVE METRICS:")
    ce = comprehensive_metrics
    print(f"  Legal Accuracy:     {ce['legal_accuracy']:.1%}")
    print(f"  Faithfulness:       {ce['faithfulness']:.1%}")
    print(f"  Completeness:       {ce['completeness']:.1%}")
    print(f"  Practicality:       {ce['practicality']:.1%}")
    print(f"  Clarity:            {ce['clarity']:.1%}")
    if ce['f2'] is not None:
        print(f"  Precision:          {ce['precision']:.1%}")
        print(f"  Recall:             {ce['recall']:.1%}")
        print(f"  F2:                 {ce['f2']:.1%}")
        print(f"  MRR:                {ce['mrr']:.4f}")
    print(f"summary_path={summary_path}")
    print(f"output_path={export_path or ''}")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    logger.log_progress(
        f"DONE run_id={run_id} total_questions={total_questions} elapsed={total_elapsed / 60:.1f}m",
        run_id=run_id,
        total_questions=total_questions,
        elapsed_seconds=round(total_elapsed, 2),
        summary_path=str(summary_path),
        output_path=str(export_path or ""),
    )

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
