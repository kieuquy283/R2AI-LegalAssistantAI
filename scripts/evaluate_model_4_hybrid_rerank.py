from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from rag.evaluation.metrics import compute_retrieval_metrics
from rag.modules.reranking import NoReranker, Reranker
from rag.modules.retrieval import HybridRetriever
from rag.retrieval.vectorstore import load_vectorstore
from rag.utils.io import load_json, save_json


MODEL_NAME = "model_4_hybrid_rerank"
MODEL_DESCRIPTION = (
    "Hybrid retrieval with dense FAISS + sparse BM25, followed by reranking over retrieved candidates. "
    "No rewrite, no history selection, no multi-query, no answer generation."
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


def _build_reranker(
    rerank_mode: str,
    candidate_k: int,
    top_k: int,
    reranker_model: str | None = None,
):
    if rerank_mode == "none":
        return NoReranker()

    kwargs: Dict[str, Any] = {
        "candidate_top_k": candidate_k,
        "output_top_k": top_k,
    }
    if reranker_model:
        kwargs["model_name"] = reranker_model
    return Reranker(**kwargs)


def evaluate_model_4_hybrid_rerank(
    eval_path: str = "data/multiturn_evaluation_legal.json",
    index_dir: str = "indexes/legal",
    corpus_path: str = "data/legal_corpus_chunks.json",
    top_k: int = 10,
    candidate_k: int = 30,
    output_path: str = "logs/eval_runs/model_4_hybrid_rerank_legal.json",
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    rerank_mode: str = "cross_encoder",
    reranker_model: str | None = None,
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

    retriever = HybridRetriever(
        vectorstore=vectorstore,
        documents=documents,
        top_k=candidate_k,
        candidate_k=candidate_k,
        fusion_type="weighted",
        alpha=normalized_dense_weight,
        filter_active=True,
    )

    reranker_init_failed = False
    reranker_init_error = None
    try:
        reranker = _build_reranker(
            rerank_mode=rerank_mode,
            candidate_k=candidate_k,
            top_k=top_k,
            reranker_model=reranker_model,
        )
    except Exception as exc:
        reranker = NoReranker()
        reranker_init_failed = True
        reranker_init_error = str(exc)

    total_hit = 0
    total_recall = 0.0
    total_mrr = 0.0
    total_latency = 0.0
    total_retrieval_latency = 0.0
    total_rerank_latency = 0.0
    evaluated_samples = 0
    sample_results: List[Dict[str, Any]] = []

    for index, sample in enumerate(data, start=1):
        question = str(sample.get("question") or sample.get("current_question") or "").strip()
        ground_truth_cids = list(sample.get("ground_truth_cids", []) or [])

        if not question or not ground_truth_cids:
            continue

        started_at = time.perf_counter()

        retrieval_started_at = time.perf_counter()
        retrieval_state = retriever.run(
            {
                "question": question,
                "query": question,
            }
        )
        candidate_results = list(retrieval_state.get("retrieval_results", []) or [])
        retrieval_latency_seconds = time.perf_counter() - retrieval_started_at
        candidate_cids = _extract_cids_from_results(candidate_results)

        rerank_started_at = time.perf_counter()
        rerank_failed = reranker_init_failed
        rerank_failure_reason = reranker_init_error

        try:
            rerank_state = reranker.run(
                {
                    "question": question,
                    "query": question,
                    "retrieval_results": candidate_results,
                }
            )
            reranked_results = list(rerank_state.get("reranked_results", []) or [])
            if not reranked_results:
                reranked_results = candidate_results[:top_k]
        except Exception as exc:
            rerank_failed = True
            rerank_failure_reason = str(exc)
            reranked_results = candidate_results[:top_k]

        if rerank_mode == "none":
            reranked_results = reranked_results[:top_k]

        rerank_latency_seconds = time.perf_counter() - rerank_started_at
        retrieved_cids_after_rerank = _extract_cids_from_results(reranked_results[:top_k])
        latency_seconds = time.perf_counter() - started_at

        hit, recall, mrr = compute_retrieval_metrics(
            retrieved_cids_after_rerank,
            ground_truth_cids,
        )

        evaluated_samples += 1
        total_hit += hit
        total_recall += recall
        total_mrr += mrr
        total_latency += latency_seconds
        total_retrieval_latency += retrieval_latency_seconds
        total_rerank_latency += rerank_latency_seconds

        sample_results.append(
            {
                "sample_id": sample.get("sample_id", sample.get("id", index)),
                "question": question,
                "query_used": question,
                "ground_truth_cids": ground_truth_cids,
                "candidate_cids_before_rerank": candidate_cids,
                "retrieved_cids_after_rerank": retrieved_cids_after_rerank,
                "hit": hit,
                "recall": recall,
                "mrr": mrr,
                "latency_seconds": latency_seconds,
                "retrieval_latency_seconds": retrieval_latency_seconds,
                "rerank_latency_seconds": rerank_latency_seconds,
                "rerank_failed": rerank_failed,
                "rerank_failure_reason": rerank_failure_reason,
            }
        )

    average_latency = 0.0
    average_retrieval_latency = 0.0
    average_rerank_latency = 0.0
    hit_at_k = 0.0
    recall_at_k = 0.0
    mrr_value = 0.0
    if evaluated_samples > 0:
        average_latency = total_latency / evaluated_samples
        average_retrieval_latency = total_retrieval_latency / evaluated_samples
        average_rerank_latency = total_rerank_latency / evaluated_samples
        hit_at_k = total_hit / evaluated_samples
        recall_at_k = total_recall / evaluated_samples
        mrr_value = total_mrr / evaluated_samples

    report = {
        "metrics": {
            "model_name": MODEL_NAME,
            "description": MODEL_DESCRIPTION,
            "eval_path": eval_path,
            "index_dir": index_dir,
            "corpus_path": corpus_path,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "dense_weight": normalized_dense_weight,
            "sparse_weight": normalized_sparse_weight,
            "rerank_mode": rerank_mode,
            "reranker_model": reranker_model,
            "samples": evaluated_samples,
            f"hit@{top_k}": hit_at_k,
            f"recall@{top_k}": recall_at_k,
            "mrr": mrr_value,
            "avg_latency_seconds": average_latency,
            "avg_retrieval_latency_seconds": average_retrieval_latency,
            "avg_rerank_latency_seconds": average_rerank_latency,
        },
        "samples": sample_results,
    }

    save_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Model 4: hybrid retrieval followed by reranking.",
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
        "--candidate-k",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--output-path",
        default="logs/eval_runs/model_4_hybrid_rerank_legal.json",
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
        "--reranker-model",
        default=None,
    )
    parser.add_argument(
        "--rerank-mode",
        choices=["cross_encoder", "none"],
        default="cross_encoder",
    )
    args = parser.parse_args()

    report = evaluate_model_4_hybrid_rerank(
        eval_path=args.eval_path,
        index_dir=args.index_dir,
        corpus_path=args.corpus_path,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        output_path=args.output_path,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        rerank_mode=args.rerank_mode,
        reranker_model=args.reranker_model,
    )

    metrics = report["metrics"]
    current_top_k = metrics["top_k"]
    print("===== MODEL 4 HYBRID RETRANK =====")
    print(f"Samples: {metrics['samples']}")
    print(f"Hit@{current_top_k}: {metrics[f'hit@{current_top_k}']:.4f}")
    print(f"Recall@{current_top_k}: {metrics[f'recall@{current_top_k}']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Avg latency (s): {metrics['avg_latency_seconds']:.4f}")
    print(f"Avg retrieval latency (s): {metrics['avg_retrieval_latency_seconds']:.4f}")
    print(f"Avg rerank latency (s): {metrics['avg_rerank_latency_seconds']:.4f}")


if __name__ == "__main__":
    main()
