from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag.utils.io import load_json, save_json


REQUIRED_FIELDS = (
    "current_question",
    "conversation",
    "gold_context_id",
    "context",
    "chunk_id",
)


def _normalize_conversation(conversation: Any) -> List[Dict[str, str]]:
    if not isinstance(conversation, list):
        return []

    normalized: List[Dict[str, str]] = []
    for item in conversation:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip() or "user"
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        normalized.append(
            {
                "role": role,
                "text": text,
                "content": text,
            }
        )

    return normalized


def _validate_sample(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in sample or sample.get(field) in (None, "")
    ]

    if "gold_context_id" not in missing_fields and not isinstance(sample.get("gold_context_id"), list):
        missing_fields.append("gold_context_id")

    if "conversation" not in missing_fields and not isinstance(sample.get("conversation"), list):
        missing_fields.append("conversation")

    return len(missing_fields) == 0, missing_fields


def prepare_legal_dataset(
    input_path: str = "data/Legal_Dataset_V1.json",
    corpus_output: str = "data/legal_corpus_chunks.json",
    eval_output: str = "data/multiturn_evaluation_legal.json",
) -> Dict[str, Any]:
    raw_data = load_json(input_path, [])
    if not isinstance(raw_data, list):
        raise ValueError("Input dataset must be a JSON list.")

    corpus_by_chunk_id: Dict[str, Dict[str, Any]] = {}
    evaluation_samples: List[Dict[str, Any]] = []

    raw_samples = len(raw_data)
    skipped_samples = 0

    for index, sample in enumerate(raw_data, start=1):
        if not isinstance(sample, dict):
            skipped_samples += 1
            print(f"[WARN] Skipping sample at index {index}: sample is not an object.")
            continue

        is_valid, missing_fields = _validate_sample(sample)
        sample_id = sample.get("sample_id", f"sample_{index}")

        if not is_valid:
            skipped_samples += 1
            print(
                f"[WARN] Skipping sample '{sample_id}': missing/invalid fields = {missing_fields}"
            )
            continue

        question = str(sample.get("current_question", "")).strip()
        context = str(sample.get("context", "")).strip()
        chunk_id = str(sample.get("chunk_id", "")).strip()
        doc_id = str(sample.get("doc_id", "")).strip()
        gold_context_id = [
            str(item).strip()
            for item in sample.get("gold_context_id", [])
            if str(item).strip()
        ]
        gold_answer = str(sample.get("gold_answer", "")).strip()
        conversation = _normalize_conversation(sample.get("conversation", []))

        if not question or not context or not chunk_id or not gold_context_id:
            skipped_samples += 1
            print(
                f"[WARN] Skipping sample '{sample_id}': empty question/context/chunk_id/gold_context_id after normalization."
            )
            continue

        evaluation_samples.append(
            {
                "sample_id": sample_id,
                "question": question,
                "current_question": question,
                "history": conversation,
                "conversation": conversation,
                "ground_truth_cids": gold_context_id,
                "gold_context_id": gold_context_id,
                "gold_answer": gold_answer,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
            }
        )

        if chunk_id not in corpus_by_chunk_id:
            corpus_by_chunk_id[chunk_id] = {
                "cid": chunk_id,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "content": context,
                "text": context,
                "metadata": {
                    "cid": chunk_id,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "sample_ids": [sample_id],
                },
            }
        else:
            existing = corpus_by_chunk_id[chunk_id]
            sample_ids = existing["metadata"].setdefault("sample_ids", [])
            if sample_id not in sample_ids:
                sample_ids.append(sample_id)

    corpus_chunks = list(corpus_by_chunk_id.values())

    save_json(corpus_output, corpus_chunks)
    save_json(eval_output, evaluation_samples)

    summary = {
        "raw_samples": raw_samples,
        "valid_evaluation_samples": len(evaluation_samples),
        "unique_corpus_chunks": len(corpus_chunks),
        "skipped_samples": skipped_samples,
        "corpus_output": corpus_output,
        "eval_output": eval_output,
    }

    print(f"Raw samples: {summary['raw_samples']}")
    print(f"Valid evaluation samples: {summary['valid_evaluation_samples']}")
    print(f"Unique corpus chunks: {summary['unique_corpus_chunks']}")
    print(f"Corpus output: {summary['corpus_output']}")
    print(f"Evaluation output: {summary['eval_output']}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Legal_Dataset_V1.json for Multi-turnRAG evaluation and indexing.",
    )
    parser.add_argument(
        "--input-path",
        default="data/Legal_Dataset_V1.json",
    )
    parser.add_argument(
        "--corpus-output",
        default="data/legal_corpus_chunks.json",
    )
    parser.add_argument(
        "--eval-output",
        default="data/multiturn_evaluation_legal.json",
    )
    args = parser.parse_args()

    prepare_legal_dataset(
        input_path=args.input_path,
        corpus_output=args.corpus_output,
        eval_output=args.eval_output,
    )


if __name__ == "__main__":
    main()
