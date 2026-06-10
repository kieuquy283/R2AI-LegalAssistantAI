from __future__ import annotations

import argparse
import json
import os
import re
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


def normalize_answer(text: str) -> str:
    cleaned = str(text or "").replace("**", " ")
    cleaned = re.sub(r"\r?\n+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return cleaned

    numbered_sections = re.split(r"(?=\b[1-5]\.\s*)", cleaned)
    if sum(1 for section in numbered_sections if re.match(r"^\s*[1-5]\.\s*", section)) >= 2:
        kept_sections: list[str] = []
        for section in numbered_sections:
            match = re.match(r"^\s*([1-5])\.\s*(.*)$", section)
            if not match:
                continue
            if match.group(1) == "2":
                continue
            body = re.sub(r"^[^:]{0,80}:\s*", "", match.group(2)).strip()
            if body:
                kept_sections.append(body)
        if kept_sections:
            return re.sub(r"\s+", " ", " ".join(kept_sections)).strip(" ,;:-")

    heading_patterns = [
        r"\b\d+\.\s*Kết luận ngắn\s*:\s*",
        r"\b\d+\.\s*Phân tích áp dụng vào tình huống\s*:\s*",
        r"\b\d+\.\s*Việc SME nên làm\s*:\s*",
        r"\b\d+\.\s*Lưu ý/rủi ro\s*:\s*",
        r"\bKết luận ngắn\s*:\s*",
        r"\bPhân tích áp dụng vào tình huống\s*:\s*",
        r"\bViệc SME nên làm\s*:\s*",
        r"\bLưu ý/rủi ro\s*:\s*",
    ]
    for pattern in heading_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(
        r"\b\d+\.\s*Căn cứ pháp luật\s*:.*?(?=(\b\d+\.\s*[A-ZÀ-Ỹa-zà-ỹ]|$))",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bCăn cứ pháp luật\s*:.*$", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d+\.\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned


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
            answer=normalize_answer(str(qa_result.get("answer") or "")),
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
