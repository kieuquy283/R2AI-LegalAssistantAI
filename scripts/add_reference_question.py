from __future__ import annotations

from typing import Any, Dict, List

from rag.utils.io import load_json, save_json


def get_last_user_question(history: List[Dict[str, Any]]) -> str:
    if not history:
        return ""

    for msg in reversed(history):
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip()

    return ""


def build_reference_question(last_user_q: str, follow_up_q: str) -> str:
    if not last_user_q:
        return follow_up_q.strip()
    return f"{last_user_q} {follow_up_q}".strip()


def add_reference_question(
    input_path: str = "data/multiturn_evaluation.json",
    output_path: str = "data/multiturn_evaluation_with_ref.json",
) -> None:
    data = load_json(input_path, [])
    updated = 0

    for sample in data:
        history = sample.get("history", [])
        question = str(sample.get("question", "")).strip()
        if not question:
            continue

        last_user_q = get_last_user_question(history)
        sample["reference_question"] = build_reference_question(last_user_q, question)
        updated += 1

    save_json(output_path, data)
    print(f"[DONE] Added reference_question for {updated} samples")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    add_reference_question()
