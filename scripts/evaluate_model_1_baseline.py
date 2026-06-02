from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag.evaluation.metrics import compute_retrieval_metrics
from rag.retrieval.ranking import filter_active_docs
from rag.retrieval.retriever import extract_cids_from_docs, retrieve_documents
from rag.retrieval.vectorstore import load_vectorstore
from rag.utils.io import load_json, save_json


MODEL_NAME = "model_1_baseline_dense_faiss"
MODEL_DESCRIPTION = (
    "Baseline using original question and FAISS dense retrieval only. "
    "No rewrite, no history selection, no hybrid retrieval, no multi-query, no reranking."
)


def _validate_index_dir(index_dir: str) -> None:
    index_path = Path(index_dir)
    if index_path.exists():
        return

    fallback_path = Path("faiss_index")
    if index_path == Path("indexes/default") and fallback_path.exists():
        raise FileNotFoundError(
            "Khong tim thay thu muc index 'indexes/default'. "
            "Neu repo cua ban dang dung index cu, hay thu: --index-dir faiss_index"
        )

    raise FileNotFoundError(f"Khong tim thay thu muc index: {index_dir}")


def evaluate_model_1_baseline(
    eval_path: str = "data/multiturn_evaluation_filled.json",
    index_dir: str = "indexes/default",
    top_k: int = 10,
    output_path: str = "logs/eval_runs/model_1_baseline.json",
) -> Dict[str, Any]:
    data = load_json(eval_path, [])
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset phai la mot list JSON.")

    _validate_index_dir(index_dir)
    vectorstore = load_vectorstore(index_dir=index_dir)

    total_hit = 0
    total_recall = 0.0
    total_mrr = 0.0
    total_latency = 0.0
    evaluated_samples = 0
    sample_results: List[Dict[str, Any]] = []

    for index, sample in enumerate(data, start=1):
        question = str(sample.get("question", "")).strip()
        ground_truth_cids = list(sample.get("ground_truth_cids", []) or [])

        if not question or not ground_truth_cids:
            continue

        query_used = question

        started_at = time.perf_counter()
        docs = retrieve_documents(
            query=query_used,
            vectorstore=vectorstore,
            top_k=top_k,
        )
        docs = filter_active_docs(docs, top_k=top_k)
        retrieved_cids = extract_cids_from_docs(docs)
        latency_seconds = time.perf_counter() - started_at

        hit, recall, mrr = compute_retrieval_metrics(retrieved_cids, ground_truth_cids)

        evaluated_samples += 1
        total_hit += hit
        total_recall += recall
        total_mrr += mrr
        total_latency += latency_seconds

        sample_results.append(
            {
                "sample_id": sample.get("sample_id", sample.get("id", index)),
                "question": question,
                "query_used": query_used,
                "ground_truth_cids": ground_truth_cids,
                "retrieved_cids": retrieved_cids,
                "hit": hit,
                "recall": recall,
                "mrr": mrr,
                "latency_seconds": latency_seconds,
            }
        )

    average_latency = 0.0
    hit_at_k = 0.0
    recall_at_k = 0.0
    mrr_value = 0.0

    if evaluated_samples > 0:
        average_latency = total_latency / evaluated_samples
        hit_at_k = total_hit / evaluated_samples
        recall_at_k = total_recall / evaluated_samples
        mrr_value = total_mrr / evaluated_samples

    report = {
        "metrics": {
            "model_name": MODEL_NAME,
            "description": MODEL_DESCRIPTION,
            "eval_path": eval_path,
            "index_dir": index_dir,
            "top_k": top_k,
            "samples": evaluated_samples,
            f"hit@{top_k}": hit_at_k,
            f"recall@{top_k}": recall_at_k,
            "mrr": mrr_value,
            "avg_latency_seconds": average_latency,
        },
        "samples": sample_results,
    }

    save_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Model 1 baseline: original question + FAISS dense retrieval only.",
    )
    parser.add_argument(
        "--eval-path",
        default="data/multiturn_evaluation_filled.json",
    )
    parser.add_argument(
        "--index-dir",
        default="indexes/default",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output-path",
        default="logs/eval_runs/model_1_baseline.json",
    )
    args = parser.parse_args()

    report = evaluate_model_1_baseline(
        eval_path=args.eval_path,
        index_dir=args.index_dir,
        top_k=args.top_k,
        output_path=args.output_path,
    )

    metrics = report["metrics"]
    top_k = metrics["top_k"]
    print("===== MODEL 1 BASELINE DENSE FAISS =====")
    print(f"Samples: {metrics['samples']}")
    print(f"Hit@{top_k}: {metrics[f'hit@{top_k}']:.4f}")
    print(f"Recall@{top_k}: {metrics[f'recall@{top_k}']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Avg latency (s): {metrics['avg_latency_seconds']:.4f}")


if __name__ == "__main__":
    main()
