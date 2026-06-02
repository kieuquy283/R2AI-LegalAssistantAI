from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag.evaluation.metrics import compute_retrieval_metrics
from rag.modules.query_rewriting import LLMQueryRewrite
from rag.retrieval.ranking import filter_active_docs
from rag.retrieval.retriever import extract_cids_from_docs, retrieve_documents
from rag.retrieval.vectorstore import load_vectorstore
from rag.utils.io import load_json, save_json


MODEL_NAME = "model_2_rewrite_dense_faiss"
MODEL_DESCRIPTION = (
    "Query rewriting with conversation history, followed by FAISS dense retrieval. "
    "No hybrid retrieval, no multi-query, no reranking."
)


def _validate_index_dir(index_dir: str) -> None:
    index_path = Path(index_dir)
    if index_path.exists():
        return

    fallback_path = Path("faiss_index")
    if index_path == Path("indexes/default") and fallback_path.exists():
        raise FileNotFoundError(
            "Index path not found. If your local index is faiss_index, run with --index-dir faiss_index"
        )

    raise FileNotFoundError(f"Khong tim thay thu muc index: {index_dir}")


def get_sample_history(
    sample: Dict[str, Any],
    max_history_turns: int = 6,
) -> List[Dict[str, Any]]:
    raw_history = (
        sample.get("history")
        or sample.get("conversation_history")
        or sample.get("turns")
        or sample.get("messages")
        or []
    )

    if not isinstance(raw_history, list):
        return []

    normalized_history: List[Dict[str, Any]] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue

        role = str(
            item.get("role")
            or item.get("speaker")
            or item.get("author")
            or "user"
        ).strip()

        content = str(
            item.get("content")
            or item.get("text")
            or item.get("message")
            or ""
        ).strip()

        if not content:
            continue

        normalized_history.append(
            {
                "role": role,
                "content": content,
            }
        )

    if max_history_turns > 0:
        return normalized_history[-max_history_turns:]
    return normalized_history


def rewrite_query_for_model_2(
    question: str,
    history: List[Dict[str, Any]],
    rewrite_mode: str = "llm",
    rewriter: LLMQueryRewrite | None = None,
) -> Tuple[str, bool, bool, float]:
    started_at = time.perf_counter()

    if rewrite_mode == "none":
        return question, False, False, time.perf_counter() - started_at

    if not history:
        return question, False, False, time.perf_counter() - started_at

    if rewriter is None:
        raise ValueError("LLM rewriter is required when rewrite_mode='llm'.")

    try:
        state = {
            "question": question,
            "selected_history": history,
        }
        rewritten_state = rewriter.run(state)
        rewritten_query = str(rewritten_state.get("rewritten_query", "")).strip()

        if not rewritten_query:
            return question, False, True, time.perf_counter() - started_at

        return rewritten_query, True, False, time.perf_counter() - started_at
    except Exception:
        return question, False, True, time.perf_counter() - started_at


def evaluate_model_2_rewrite_dense(
    eval_path: str = "data/multiturn_evaluation_filled.json",
    index_dir: str = "indexes/default",
    top_k: int = 10,
    output_path: str = "logs/eval_runs/model_2_rewrite_dense.json",
    rewrite_mode: str = "llm",
    max_history_turns: int = 6,
) -> Dict[str, Any]:
    data = load_json(eval_path, [])
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset phai la mot list JSON.")

    _validate_index_dir(index_dir)
    vectorstore = load_vectorstore(index_dir=index_dir)

    rewriter = None
    if rewrite_mode == "llm":
        rewriter = LLMQueryRewrite()

    total_hit = 0
    total_recall = 0.0
    total_mrr = 0.0
    total_latency = 0.0
    total_rewrite_latency = 0.0
    rewrite_success_count = 0
    evaluated_samples = 0
    sample_results: List[Dict[str, Any]] = []

    for index, sample in enumerate(data, start=1):
        question = str(sample.get("question", "")).strip()
        ground_truth_cids = list(sample.get("ground_truth_cids", []) or [])

        if not question or not ground_truth_cids:
            continue

        history = get_sample_history(sample, max_history_turns=max_history_turns)
        rewritten_query, rewrite_used, rewrite_failed, rewrite_latency_seconds = rewrite_query_for_model_2(
            question=question,
            history=history,
            rewrite_mode=rewrite_mode,
            rewriter=rewriter,
        )
        query_used = rewritten_query or question

        started_at = time.perf_counter()
        docs = retrieve_documents(
            query=query_used,
            vectorstore=vectorstore,
            top_k=top_k,
        )
        docs = filter_active_docs(docs, top_k=top_k)
        retrieved_cids = extract_cids_from_docs(docs)
        retrieval_latency_seconds = time.perf_counter() - started_at
        latency_seconds = rewrite_latency_seconds + retrieval_latency_seconds

        hit, recall, mrr = compute_retrieval_metrics(retrieved_cids, ground_truth_cids)

        evaluated_samples += 1
        total_hit += hit
        total_recall += recall
        total_mrr += mrr
        total_latency += latency_seconds
        total_rewrite_latency += rewrite_latency_seconds
        if rewrite_used and not rewrite_failed:
            rewrite_success_count += 1

        sample_results.append(
            {
                "sample_id": sample.get("sample_id", sample.get("id", index)),
                "question": question,
                "history": history,
                "original_question": question,
                "rewritten_query": rewritten_query,
                "query_used": query_used,
                "rewrite_used": rewrite_used,
                "rewrite_failed": rewrite_failed,
                "ground_truth_cids": ground_truth_cids,
                "retrieved_cids": retrieved_cids,
                "hit": hit,
                "recall": recall,
                "mrr": mrr,
                "latency_seconds": latency_seconds,
                "rewrite_latency_seconds": rewrite_latency_seconds,
            }
        )

    average_latency = 0.0
    average_rewrite_latency = 0.0
    hit_at_k = 0.0
    recall_at_k = 0.0
    mrr_value = 0.0
    rewrite_success_rate = 0.0

    if evaluated_samples > 0:
        average_latency = total_latency / evaluated_samples
        average_rewrite_latency = total_rewrite_latency / evaluated_samples
        hit_at_k = total_hit / evaluated_samples
        recall_at_k = total_recall / evaluated_samples
        mrr_value = total_mrr / evaluated_samples
        rewrite_success_rate = rewrite_success_count / evaluated_samples

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
            "avg_rewrite_latency_seconds": average_rewrite_latency,
            "rewrite_success_rate": rewrite_success_rate,
            "rewrite_mode": rewrite_mode,
            "max_history_turns": max_history_turns,
        },
        "samples": sample_results,
    }

    save_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Model 2: query rewriting + FAISS dense retrieval.",
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
        default="logs/eval_runs/model_2_rewrite_dense.json",
    )
    parser.add_argument(
        "--rewrite-mode",
        default="llm",
        choices=["llm", "none"],
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=6,
    )
    args = parser.parse_args()

    report = evaluate_model_2_rewrite_dense(
        eval_path=args.eval_path,
        index_dir=args.index_dir,
        top_k=args.top_k,
        output_path=args.output_path,
        rewrite_mode=args.rewrite_mode,
        max_history_turns=args.max_history_turns,
    )

    metrics = report["metrics"]
    top_k = metrics["top_k"]
    print("===== MODEL 2 REWRITE + DENSE FAISS =====")
    print(f"Samples: {metrics['samples']}")
    print(f"Hit@{top_k}: {metrics[f'hit@{top_k}']:.4f}")
    print(f"Recall@{top_k}: {metrics[f'recall@{top_k}']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Avg latency (s): {metrics['avg_latency_seconds']:.4f}")
    print(f"Avg rewrite latency (s): {metrics['avg_rewrite_latency_seconds']:.4f}")
    print(f"Rewrite success rate: {metrics['rewrite_success_rate']:.4f}")
    print(json.dumps({"output_path": args.output_path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
