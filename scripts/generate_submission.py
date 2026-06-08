from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.qa_pipeline import LegalQAPipeline
from legal_rag.submission import SubmissionItem, export_submission


def load_questions(path: str | Path) -> list[dict]:
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

    rows: list[dict] = []
    with question_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


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


def parse_question(raw_item: dict, fallback_id: int) -> tuple[int, str]:
    question_id = None
    for key in ("id", "question_id", "sample_id", "qid"):
        question_id = _coerce_int_like(raw_item.get(key))
        if question_id is not None:
            break
    if question_id is None:
        question_id = fallback_id
    question = str(raw_item.get("question") or raw_item.get("current_question") or "").strip()
    if not question:
        raise ValueError(f"Question item {question_id} is missing question text.")
    return question_id, question


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate competition submission results.json.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ["EMBEDDING_BACKEND"] = "hash"

    questions = load_questions(args.input)
    qa = LegalQAPipeline()

    def _run_item(payload: tuple[int, dict]) -> SubmissionItem:
        index, raw_item = payload
        question_id, question = parse_question(raw_item, index)
        qa_result = qa.answer(question, include_grounding=False, use_llm=False)
        return SubmissionItem(
            id=question_id,
            question=question,
            answer=str(qa_result.get("answer") or ""),
            relevant_docs=list(dict.fromkeys(qa_result.get("relevant_docs") or [])),
            relevant_articles=list(dict.fromkeys(qa_result.get("relevant_articles") or [])),
        )

    worker_count = max(1, min(int(os.getenv("R2AI_SUBMISSION_WORKERS", "4")), 8))
    if worker_count == 1:
        submission_items = [_run_item((index, raw_item)) for index, raw_item in enumerate(questions, start=1)]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            submission_items = list(executor.map(_run_item, enumerate(questions, start=1)))

    export_submission(submission_items, args.output)
    print(f"Generated {len(submission_items)} submission items at {Path(args.output)}")


if __name__ == "__main__":
    main()
