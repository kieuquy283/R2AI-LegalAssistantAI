from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from rag.evaluation.metrics import compute_retrieval_metrics
from rag.modules.retrieval import HybridRetriever
from rag.retrieval.vectorstore import load_vectorstore
from rag.utils.io import load_json, save_json


MODEL_NAME = "model_3_hybrid_retrieval"
MODEL_DESCRIPTION = (
    "Hybrid retrieval using the original question with dense FAISS + sparse BM25 fusion. "
    "No rewrite, no history selection, no multi-query, no reranking, no answer generation."
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


def _normalize_weights(dense_weight: float, sparse_weight: float) -> Tuple[float, float, float]:
    dense = float(dense_weight)
    sparse = float(sparse_weight)
    total = dense + sparse
    if total <= 0:
        raise ValueError("dense_weight + sparse_weight phai lon hon 0.")
    normalized_dense = dense / total
    normalized_sparse = sparse / total
    return normalized_dense, normalized_sparse, total


def evaluate_model_3_hybrid(
    eval_path: str = "data/multiturn_evaluation_legal.json",
    index_dir: str = "indexes/legal",
    corpus_path: str = "data/legal_corpus_chunks.json",
    top_k: int = 10,
    output_path: str = "logs/eval_runs/model_3_hybrid_legal.json",
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> Dict[str, Any]:
    data = load_json(eval_path, [])
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset phai la mot list JSON.")

    _validate_index_dir(index_dir)
    vectorstore = load_vectorstore(index_dir=index_dir)
    documents = _load_corpus_documents(corpus_path=corpus_path)
    alpha, normalized_sparse_weight, _ = _normalize_weights(
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    retriever = HybridRetriever(
        vectorstore=vectorstore,
        documents=documents,
        top_k=top_k,
        candidate_k=max(top_k * 2, top_k + 5),
        fusion_type="weighted",
        alpha=alpha,
        filter_active=True,
    )

    total_hit = 0
    total_recall = 0.0
    total_mrr = 0.0
    total_latency = 0.0
    evaluated_samples = 0
    sample_results: List[Dict[str, Any]] = []

    for index, sample in enumerate(data, start=1):
        question = str(sample.get("question") or sample.get("current_question") or "").strip()
        ground_truth_cids = list(sample.get("ground_truth_cids", []) or [])

        if not question or not ground_truth_cids:
            continue

        started_at = time.perf_counter()
        state = retriever.run(
            {
                "question": question,
                "query": question,
            }
        )
        results = list(state.get("retrieval_results", []) or [])
        retrieved_cids = _extract_cids_from_results(results)
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
                "query_used": question,
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
            "corpus_path": corpus_path,
            "top_k": top_k,
            "dense_weight": alpha,
            "sparse_weight": normalized_sparse_weight,
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
        description="Evaluate Model 3: hybrid retrieval with dense FAISS + sparse BM25.",
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
        "--output-path",
        default="logs/eval_runs/model_3_hybrid_legal.json",
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
    args = parser.parse_args()

    report = evaluate_model_3_hybrid(
        eval_path=args.eval_path,
        index_dir=args.index_dir,
        corpus_path=args.corpus_path,
        top_k=args.top_k,
        output_path=args.output_path,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )

    metrics = report["metrics"]
    current_top_k = metrics["top_k"]
    print("===== MODEL 3 HYBRID RETRIEVAL =====")
    print(f"Samples: {metrics['samples']}")
    print(f"Hit@{current_top_k}: {metrics[f'hit@{current_top_k}']:.4f}")
    print(f"Recall@{current_top_k}: {metrics[f'recall@{current_top_k}']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Avg latency (s): {metrics['avg_latency_seconds']:.4f}")


if __name__ == "__main__":
    main()
