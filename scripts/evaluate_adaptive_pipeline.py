from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from rag.evaluation.metrics import compute_retrieval_metrics
from rag.pipelines.adaptive_modular_pipeline import AdaptiveModularPipeline
from rag.utils.io import load_json, save_json


MODEL_NAME = "adaptive_modular_pipeline"
MODEL_DESCRIPTION = (
    "Production adaptive modular retrieval runtime with progressive escalation from dense retrieval "
    "to rewrite, hybrid retrieval, reranking, full modular retrieval, and HyDE."
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


def _normalize_history(sample: Dict[str, Any], max_history_turns: int = 8) -> List[Dict[str, Any]]:
    raw_history = (
        sample.get("history")
        or sample.get("conversation_history")
        or sample.get("conversation")
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
        role = str(item.get("role") or item.get("speaker") or item.get("author") or "user").strip()
        content = str(item.get("content") or item.get("text") or item.get("message") or "").strip()
        if not content:
            continue
        normalized_history.append({"role": role, "content": content})

    if max_history_turns > 0:
        return normalized_history[-max_history_turns:]
    return normalized_history


def evaluate_adaptive_pipeline(
    *,
    eval_path: str,
    index_dir: str,
    corpus_path: str,
    top_k: int = 5,
    output_path: str = "logs/eval_runs/adaptive_pipeline_legal_top5.json",
    candidate_k: int = 40,
    history_top_k: int = 4,
    max_history_turns: int = 8,
) -> Dict[str, Any]:
    _validate_index_dir(index_dir)
    data = load_json(eval_path, [])
    if not isinstance(data, list) or not data:
        raise ValueError(f"Evaluation data is missing or invalid: {eval_path}")

    pipeline = AdaptiveModularPipeline(
        index_dir=index_dir,
        corpus_path=corpus_path,
        top_k=top_k,
        candidate_k=candidate_k,
        history_top_k=history_top_k,
        max_history_turns=max_history_turns,
    )

    total = 0
    total_hit = 0.0
    total_recall = 0.0
    total_mrr = 0.0
    total_latency = 0.0
    escalated_count = 0
    llm_count = 0
    rerank_count = 0
    hyde_count = 0
    route_distribution: Dict[str, int] = {}
    samples: List[Dict[str, Any]] = []

    for index, sample in enumerate(data, start=1):
        if not isinstance(sample, dict):
            continue
        question = str(sample.get("question", "")).strip()
        ground_truth_cids = list(sample.get("ground_truth_cids", []) or [])
        history = _normalize_history(sample, max_history_turns=max_history_turns)
        if not question or not ground_truth_cids:
            continue

        started_at = time.perf_counter()
        state = pipeline.run_retrieval(question=question, history=history)
        latency_seconds = time.perf_counter() - started_at

        metadata = dict(state.get("adaptive_metadata", {}) or {})
        retrieved_cids = list(metadata.get("retrieved_cids", []) or [])
        hit, recall, mrr = compute_retrieval_metrics(retrieved_cids, ground_truth_cids)

        total += 1
        total_hit += float(hit)
        total_recall += float(recall)
        total_mrr += float(mrr)
        total_latency += latency_seconds

        final_route = str(metadata.get("final_route") or "unknown")
        route_distribution[final_route] = route_distribution.get(final_route, 0) + 1
        if metadata.get("escalated"):
            escalated_count += 1
        if int(metadata.get("llm_calls", 0)) > 0:
            llm_count += 1
        if metadata.get("used_rerank"):
            rerank_count += 1
        if metadata.get("used_hyde"):
            hyde_count += 1

        samples.append(
            {
                "sample_id": sample.get("sample_id") or sample.get("id") or index,
                "question": question,
                "selected_route": metadata.get("selected_route"),
                "final_route": final_route,
                "escalated": bool(metadata.get("escalated")),
                "escalation_path": list(metadata.get("escalation_path", []) or []),
                "query_used": metadata.get("query_used") or question,
                "retrieved_cids": retrieved_cids,
                "ground_truth_cids": ground_truth_cids,
                "confidence_score": float(metadata.get("confidence_score", 0.0)),
                "hit": float(hit),
                "recall": float(recall),
                "mrr": float(mrr),
                "latency_seconds": float(latency_seconds),
            }
        )

    if total == 0:
        raise ValueError(f"No valid evaluation samples found in: {eval_path}")

    metrics = {
        "model_name": MODEL_NAME,
        "description": MODEL_DESCRIPTION,
        "samples": total,
        "top_k": top_k,
        f"hit@{top_k}": total_hit / total,
        f"recall@{top_k}": total_recall / total,
        "mrr": total_mrr / total,
        "avg_latency_seconds": total_latency / total,
        "route_distribution": route_distribution,
        "escalation_rate": escalated_count / total,
        "llm_call_rate": llm_count / total,
        "rerank_rate": rerank_count / total,
        "hyde_rate": hyde_count / total,
    }

    report = {
        "model_name": MODEL_NAME,
        "eval_path": eval_path,
        "index_dir": index_dir,
        "corpus_path": corpus_path,
        "metrics": metrics,
        "samples": samples,
    }
    save_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the adaptive modular retrieval pipeline.")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--history-top-k", type=int, default=4)
    parser.add_argument("--max-history-turns", type=int, default=8)
    parser.add_argument(
        "--output-path",
        default="logs/eval_runs/adaptive_pipeline_legal_top5.json",
    )
    args = parser.parse_args()

    report = evaluate_adaptive_pipeline(
        eval_path=args.eval_path,
        index_dir=args.index_dir,
        corpus_path=args.corpus_path,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        history_top_k=args.history_top_k,
        max_history_turns=args.max_history_turns,
        output_path=args.output_path,
    )
    metrics = report["metrics"]
    current_top_k = metrics["top_k"]
    print("===== ADAPTIVE MODULAR PIPELINE =====")
    print(f"Samples: {metrics['samples']}")
    print(f"Hit@{current_top_k}: {metrics[f'hit@{current_top_k}']:.4f}")
    print(f"Recall@{current_top_k}: {metrics[f'recall@{current_top_k}']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Avg latency (s): {metrics['avg_latency_seconds']:.4f}")
    print(f"Escalation rate: {metrics['escalation_rate']:.4f}")
    print(f"LLM call rate: {metrics['llm_call_rate']:.4f}")
    print(f"Rerank rate: {metrics['rerank_rate']:.4f}")
    print(f"HyDE rate: {metrics['hyde_rate']:.4f}")
    print(f"Route distribution: {metrics['route_distribution']}")


if __name__ == "__main__":
    main()
