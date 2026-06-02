from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from rag.evaluation.metrics import compute_retrieval_metrics
from rag.modules.history_selection import HybridHistorySelector, RecencyHistorySelector
from rag.modules.query_rewriting import LLMQueryRewrite, NoRewrite
from rag.modules.retrieval import HybridRetriever
from rag.retrieval.vectorstore import get_embeddings, load_vectorstore
from rag.utils.io import load_json, save_json


MODEL_NAME = "model_5_hybrid_history_rewrite_hybrid"
MODEL_DESCRIPTION = (
    "Hybrid history selection over conversation turns, followed by query rewriting and hybrid retrieval. "
    "No reranking, no multi-query, no answer generation."
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


def _load_corpus_documents(corpus_path: str) -> List[Document]:
    corpus = load_json(corpus_path, [])
    if not isinstance(corpus, list):
        raise ValueError("Corpus JSON phai la mot list.")

    documents: List[Document] = []
    for index, item in enumerate(corpus, start=1):
        if not isinstance(item, dict):
            continue

        text = str(item.get("content") or item.get("text") or "").strip()
        if not text:
            continue

        item_metadata = dict(item.get("metadata", {}) or {})
        chunk_id = str(item.get("chunk_id") or item_metadata.get("chunk_id") or index)
        cid = str(item.get("cid") or item_metadata.get("cid") or chunk_id)
        doc_id = str(item.get("doc_id") or item_metadata.get("doc_id") or "")

        metadata = dict(item_metadata)
        metadata["cid"] = cid
        metadata["chunk_id"] = chunk_id
        metadata["doc_id"] = doc_id

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


def _extract_cids_from_results(results: List[Any]) -> List[str]:
    retrieved_cids: List[str] = []
    for item in results:
        metadata = dict(getattr(item, "metadata", {}) or {})
        chunk_id = str(getattr(item, "chunk_id", "") or "").strip()
        cid = str(metadata.get("cid") or metadata.get("chunk_id") or chunk_id).strip()
        if cid:
            retrieved_cids.append(cid)
    return retrieved_cids


def _normalize_weights(dense_weight: float, sparse_weight: float) -> Tuple[float, float]:
    dense = float(dense_weight)
    sparse = float(sparse_weight)
    total = dense + sparse
    if total <= 0:
        raise ValueError("dense_weight + sparse_weight phai lon hon 0.")
    return dense / total, sparse / total


def get_sample_history(
    sample: Dict[str, Any],
    max_history_turns: int = 8,
) -> List[Dict[str, Any]]:
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


def _build_history_selector(
    history_mode: str,
    history_top_k: int,
    max_history_turns: int,
):
    if history_mode == "recent":
        return RecencyHistorySelector(top_k=history_top_k)

    return HybridHistorySelector(
        embedding_model=get_embeddings(),
        top_k=history_top_k,
        alpha=0.8,
        beta=0.2,
        recent_window=max_history_turns,
    )


def _build_query_rewriter(rewrite_mode: str):
    if rewrite_mode == "none":
        return NoRewrite()
    return LLMQueryRewrite()


def select_history_for_model_5(
    question: str,
    history: List[Dict[str, Any]],
    selector,
    fallback_selector: RecencyHistorySelector,
) -> Tuple[List[Dict[str, Any]], bool, float]:
    started_at = time.perf_counter()

    if not history:
        return [], False, time.perf_counter() - started_at

    try:
        state = selector.run(
            {
                "question": question,
                "query": question,
                "history": history,
            }
        )
        selected_history = list(state.get("selected_history", []) or [])
        return selected_history, False, time.perf_counter() - started_at
    except Exception:
        fallback_state = fallback_selector.run(
            {
                "question": question,
                "query": question,
                "history": history,
            }
        )
        selected_history = list(fallback_state.get("selected_history", []) or [])
        return selected_history, True, time.perf_counter() - started_at


def rewrite_query_for_model_5(
    question: str,
    selected_history: List[Dict[str, Any]],
    rewrite_mode: str,
    rewriter,
) -> Tuple[str, bool, float]:
    started_at = time.perf_counter()

    if rewrite_mode == "none":
        return question, False, time.perf_counter() - started_at

    if not selected_history:
        return question, False, time.perf_counter() - started_at

    try:
        state = rewriter.run(
            {
                "question": question,
                "selected_history": selected_history,
            }
        )
        rewritten_query = str(state.get("rewritten_query", "")).strip()
        rewrite_metadata = dict(state.get("query_rewriting", {}) or {})
        rewrite_failed = False

        if not rewritten_query:
            rewritten_query = question
            rewrite_failed = True

        fallback_reason = rewrite_metadata.get("fallback_reason")
        if fallback_reason and rewritten_query == question:
            rewrite_failed = True

        return rewritten_query or question, rewrite_failed, time.perf_counter() - started_at
    except Exception:
        return question, True, time.perf_counter() - started_at


def evaluate_model_5_hybrid_history(
    eval_path: str = "data/multiturn_evaluation_legal.json",
    index_dir: str = "indexes/legal",
    corpus_path: str = "data/legal_corpus_chunks.json",
    top_k: int = 10,
    history_top_k: int = 4,
    output_path: str = "logs/eval_runs/model_5_hybrid_history_legal.json",
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    max_history_turns: int = 8,
    rewrite_mode: str = "llm",
    history_mode: str = "hybrid",
) -> Dict[str, Any]:
    data = load_json(eval_path, [])
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset phai la mot list JSON.")

    _validate_index_dir(index_dir)
    vectorstore = load_vectorstore(index_dir=index_dir)
    documents = _load_corpus_documents(corpus_path=corpus_path)
    normalized_dense_weight, normalized_sparse_weight = _normalize_weights(
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    history_selector = _build_history_selector(
        history_mode=history_mode,
        history_top_k=history_top_k,
        max_history_turns=max_history_turns,
    )
    fallback_history_selector = RecencyHistorySelector(top_k=history_top_k)
    rewriter = _build_query_rewriter(rewrite_mode=rewrite_mode)
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        documents=documents,
        top_k=top_k,
        candidate_k=max(top_k * 2, top_k + 5),
        fusion_type="weighted",
        alpha=normalized_dense_weight,
        filter_active=True,
    )

    total_hit = 0
    total_recall = 0.0
    total_mrr = 0.0
    total_latency = 0.0
    total_history_selection_latency = 0.0
    total_rewrite_latency = 0.0
    total_retrieval_latency = 0.0
    rewrite_success_count = 0
    evaluated_samples = 0
    sample_results: List[Dict[str, Any]] = []

    for index, sample in enumerate(data, start=1):
        question = str(sample.get("question") or sample.get("current_question") or "").strip()
        ground_truth_cids = list(sample.get("ground_truth_cids", []) or [])

        if not question or not ground_truth_cids:
            continue

        history = get_sample_history(sample, max_history_turns=max_history_turns)
        sample_started_at = time.perf_counter()

        selected_history, history_selection_failed, history_selection_latency_seconds = (
            select_history_for_model_5(
                question=question,
                history=history,
                selector=history_selector,
                fallback_selector=fallback_history_selector,
            )
        )

        rewritten_query, rewrite_failed, rewrite_latency_seconds = rewrite_query_for_model_5(
            question=question,
            selected_history=selected_history,
            rewrite_mode=rewrite_mode,
            rewriter=rewriter,
        )
        query_used = rewritten_query or question

        retrieval_started_at = time.perf_counter()
        retrieval_state = retriever.run(
            {
                "question": question,
                "query": query_used,
                "rewritten_query": query_used,
            }
        )
        retrieved_results = list(retrieval_state.get("retrieval_results", []) or [])
        retrieved_cids = _extract_cids_from_results(retrieved_results)
        retrieval_latency_seconds = time.perf_counter() - retrieval_started_at
        latency_seconds = time.perf_counter() - sample_started_at

        hit, recall, mrr = compute_retrieval_metrics(retrieved_cids, ground_truth_cids)

        evaluated_samples += 1
        total_hit += hit
        total_recall += recall
        total_mrr += mrr
        total_latency += latency_seconds
        total_history_selection_latency += history_selection_latency_seconds
        total_rewrite_latency += rewrite_latency_seconds
        total_retrieval_latency += retrieval_latency_seconds
        if query_used != question and not rewrite_failed:
            rewrite_success_count += 1

        sample_results.append(
            {
                "sample_id": sample.get("sample_id", sample.get("id", index)),
                "original_question": question,
                "selected_history": selected_history,
                "rewritten_query": rewritten_query,
                "query_used": query_used,
                "ground_truth_cids": ground_truth_cids,
                "retrieved_cids": retrieved_cids,
                "hit": hit,
                "recall": recall,
                "mrr": mrr,
                "history_selection_latency_seconds": history_selection_latency_seconds,
                "rewrite_latency_seconds": rewrite_latency_seconds,
                "retrieval_latency_seconds": retrieval_latency_seconds,
                "latency_seconds": latency_seconds,
                "rewrite_failed": rewrite_failed,
                "history_selection_failed": history_selection_failed,
            }
        )

    average_latency = 0.0
    average_history_selection_latency = 0.0
    average_rewrite_latency = 0.0
    average_retrieval_latency = 0.0
    hit_at_k = 0.0
    recall_at_k = 0.0
    mrr_value = 0.0
    rewrite_success_rate = 0.0

    if evaluated_samples > 0:
        average_latency = total_latency / evaluated_samples
        average_history_selection_latency = total_history_selection_latency / evaluated_samples
        average_rewrite_latency = total_rewrite_latency / evaluated_samples
        average_retrieval_latency = total_retrieval_latency / evaluated_samples
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
            "corpus_path": corpus_path,
            "top_k": top_k,
            "history_top_k": history_top_k,
            "max_history_turns": max_history_turns,
            "dense_weight": normalized_dense_weight,
            "sparse_weight": normalized_sparse_weight,
            "rewrite_mode": rewrite_mode,
            "history_mode": history_mode,
            "samples": evaluated_samples,
            f"hit@{top_k}": hit_at_k,
            f"recall@{top_k}": recall_at_k,
            "mrr": mrr_value,
            "avg_latency_seconds": average_latency,
            "avg_history_selection_latency_seconds": average_history_selection_latency,
            "avg_rewrite_latency_seconds": average_rewrite_latency,
            "avg_retrieval_latency_seconds": average_retrieval_latency,
            "rewrite_success_rate": rewrite_success_rate,
        },
        "samples": sample_results,
    }

    save_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Model 5: hybrid history selection + query rewriting + hybrid retrieval.",
    )
    parser.add_argument(
        "--eval-path",
        default="data/multiturn_evaluation_legal.json",
    )
    parser.add_argument(
        "--index-dir",
        default="indexes/legal",
    )
    parser.add_argument(
        "--corpus-path",
        default="data/legal_corpus_chunks.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--history-top-k",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output-path",
        default="logs/eval_runs/model_5_hybrid_history_legal.json",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--rewrite-mode",
        choices=["llm", "none"],
        default="llm",
    )
    parser.add_argument(
        "--history-mode",
        choices=["hybrid", "recent"],
        default="hybrid",
    )
    args = parser.parse_args()

    report = evaluate_model_5_hybrid_history(
        eval_path=args.eval_path,
        index_dir=args.index_dir,
        corpus_path=args.corpus_path,
        top_k=args.top_k,
        history_top_k=args.history_top_k,
        output_path=args.output_path,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        max_history_turns=args.max_history_turns,
        rewrite_mode=args.rewrite_mode,
        history_mode=args.history_mode,
    )

    metrics = report["metrics"]
    current_top_k = metrics["top_k"]
    print("===== MODEL 5 HYBRID HISTORY + REWRITE + HYBRID =====")
    print(f"Samples: {metrics['samples']}")
    print(f"Hit@{current_top_k}: {metrics[f'hit@{current_top_k}']:.4f}")
    print(f"Recall@{current_top_k}: {metrics[f'recall@{current_top_k}']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Avg latency (s): {metrics['avg_latency_seconds']:.4f}")
    print(f"Avg history latency (s): {metrics['avg_history_selection_latency_seconds']:.4f}")
    print(f"Avg rewrite latency (s): {metrics['avg_rewrite_latency_seconds']:.4f}")
    print(f"Avg retrieval latency (s): {metrics['avg_retrieval_latency_seconds']:.4f}")
    print(f"Rewrite success rate: {metrics['rewrite_success_rate']:.4f}")


if __name__ == "__main__":
    main()
