from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from rag.utils.io import save_json


DEFAULT_DATASET_NAME = "YuITC/Vietnamese-Legal-Documents"
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT_JSON = "data/evaluation.json"


def normalize_context_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    results: List[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                results.append(text)
    return results


def normalize_cid_list(value: Any, expected_len: int) -> List[Any]:
    if not isinstance(value, list):
        return [None] * expected_len

    normalized = list(value[:expected_len])
    if len(normalized) < expected_len:
        normalized.extend([None] * (expected_len - len(normalized)))
    return normalized


def build_evaluation_samples(
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)

    if limit is not None:
        limit = max(0, min(limit, len(ds)))
        ds = ds.select(range(limit))

    samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        question = str(row.get("question", "")).strip()
        if not question:
            continue

        context_list = normalize_context_list(row.get("context_list", []))
        cid_list = normalize_cid_list(row.get("cid", []), len(context_list))
        qid = row.get("qid")

        samples.append(
            {
                "id": idx + 1,
                "qid": qid,
                "question": question,
                "ground_truth_contexts": context_list,
                "ground_truth_cids": cid_list,
                "type": "single_turn",
                "source_dataset": dataset_name,
                "split": split,
            }
        )

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare evaluation dataset from HF legal dataset.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    samples = build_evaluation_samples(
        dataset_name=args.dataset_name,
        split=args.split,
        limit=args.limit,
    )
    if not samples:
        raise ValueError("No evaluation samples were created.")

    save_json(args.output_json, samples)
    print(f"[EVAL DATASET] Saved {len(samples)} samples to: {args.output_json}")


if __name__ == "__main__":
    main()
