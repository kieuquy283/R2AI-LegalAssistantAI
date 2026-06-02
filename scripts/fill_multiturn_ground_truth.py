from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from rag.utils.io import load_json, save_json


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def find_best_match(
    reference_question: str,
    evaluation_data: List[Dict[str, Any]],
    threshold: float = 0.75,
) -> Optional[Dict[str, Any]]:
    best_item = None
    best_score = 0.0

    for item in evaluation_data:
        q = item.get("question", "")
        score = similarity(reference_question, q)
        if score > best_score:
            best_score = score
            best_item = item

    if best_item is not None and best_score >= threshold:
        return best_item

    return None


def fill_ground_truth(
    multiturn_path: str = "data/multiturn_evaluation_with_ref.json",
    evaluation_path: str = "data/evaluation.json",
    output_path: str = "data/multiturn_evaluation_filled.json",
    threshold: float = 0.5,
) -> None:
    multiturn_data = load_json(multiturn_path, [])
    evaluation_data = load_json(evaluation_path, [])

    matched = 0
    for sample in multiturn_data:
        reference_question = str(sample.get("reference_question", "")).strip()
        if not reference_question:
            continue

        best_match = find_best_match(
            reference_question,
            evaluation_data,
            threshold=threshold,
        )
        if best_match is None:
            continue

        sample["ground_truth_cids"] = best_match.get("ground_truth_cids", [])
        sample["ground_truth_contexts"] = best_match.get("ground_truth_contexts", [])
        sample["matched_question"] = best_match.get("question", "")
        matched += 1

    save_json(output_path, multiturn_data)
    print(f"[DONE] Matched {matched}/{len(multiturn_data)} samples")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiturn-path", default="data/multiturn_evaluation_with_ref.json")
    parser.add_argument("--evaluation-path", default="data/evaluation.json")
    parser.add_argument("--output-path", default="data/multiturn_evaluation_filled.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    fill_ground_truth(
        multiturn_path=args.multiturn_path,
        evaluation_path=args.evaluation_path,
        output_path=args.output_path,
        threshold=args.threshold,
    )
