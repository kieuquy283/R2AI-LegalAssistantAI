from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from src.legal_rag.utils import load_json, write_text


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


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _normalize_conversation(raw: Any) -> list[dict[str, str]]:
    conversation: list[dict[str, str]] = []
    for item in _as_list(raw):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip() or "user"
        text = str(item.get("text") or item.get("content") or item.get("message") or "").strip()
        if not text:
            continue
        conversation.append(
            {
                "role": role,
                "text": text,
                "content": text,
            }
        )
    return conversation


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("data", "questions", "items", "samples", "records", "rows", "payload"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    return []


def _pick_question_text(sample: dict[str, Any]) -> str:
    for key in ("question", "current_question", "currentQuestion", "query", "prompt"):
        text = str(sample.get(key) or "").strip()
        if text:
            return text
    return ""


def _pick_expected_refs(sample: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    raw_refs = sample.get("expected_law_refs") or sample.get("expected_refs") or sample.get("law_refs")
    for item in _as_list(raw_refs):
        ref = str(item).strip()
        if ref:
            refs.append(ref)
    return refs


def _pick_question_id(sample: dict[str, Any], fallback_id: int) -> int:
    for key in ("id", "question_id", "sample_id", "qid"):
        value = _coerce_int_like(sample.get(key))
        if value is not None:
            return value
    return fallback_id


def prepare_r2ai_dataset(
    input_path: str | Path = "data/evaluation/R2AIStage1DATA.json",
    output_path: str | Path = "data/evaluation/r2ai_stage1_questions.jsonl",
) -> dict[str, Any]:
    payload = load_json(input_path, default=None)
    if payload is None:
        raise FileNotFoundError(f"Missing input dataset: {input_path}")

    items = _extract_items(payload)
    if not items:
        raise ValueError("Input dataset must be a JSON list or a JSON object containing a list of question items.")

    output_lines: list[str] = []
    skipped = 0

    for index, sample in enumerate(items, start=1):
        question = _pick_question_text(sample)
        if not question:
            skipped += 1
            continue

        question_id = _pick_question_id(sample, index)
        prepared: dict[str, Any] = {
            "id": question_id,
            "question": question,
        }

        original_question_id = sample.get("question_id")
        if original_question_id not in (None, ""):
            prepared["question_id"] = original_question_id

        expected_law_refs = _pick_expected_refs(sample)
        if expected_law_refs:
            prepared["expected_law_refs"] = expected_law_refs

        conversation = _normalize_conversation(
            sample.get("conversation") or sample.get("conversation_history") or sample.get("history")
        )
        if conversation:
            prepared["conversation"] = conversation
            prepared["history"] = conversation

        current_question = str(sample.get("current_question") or "").strip()
        if current_question:
            prepared["current_question"] = current_question

        output_lines.append(json.dumps(prepared, ensure_ascii=False))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_text(output_file, "\n".join(output_lines) + ("\n" if output_lines else ""))

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "items_read": len(items),
        "items_written": len(output_lines),
        "items_skipped": skipped,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Prepare R2AI Stage 1 questions JSONL from the raw dataset.")
    parser.add_argument("--input", default="data/evaluation/R2AIStage1DATA.json")
    parser.add_argument("--output", default="data/evaluation/r2ai_stage1_questions.jsonl")
    args = parser.parse_args()

    prepare_r2ai_dataset(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    _cli()
