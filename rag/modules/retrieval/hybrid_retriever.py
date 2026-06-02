from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseRetriever
from .dense_retriever import FAISSRetriever
from .fusion import reciprocal_rank_fusion, weighted_fusion
from .schemas import RetrievalResult
from .sparse_retriever import SparseRetriever
from .utils import (
    deduplicate_results,
    filter_active_results,
    filter_low_score_results,
    get_effective_score,
)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining FAISS dense retrieval and BM25 sparse retrieval.
    """

    def __init__(
        self,
        vectorstore=None,
        documents=None,
        dense_retriever: Optional[FAISSRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        top_k: int = 5,
        candidate_k: Optional[int] = None,
        fusion_type: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        filter_active: bool = True,
        filter_threshold: float | None = None,
    ) -> None:
        self.top_k = int(top_k)
        self.candidate_k = int(candidate_k or max(self.top_k * 2, self.top_k + 5))
        self.fusion_type = fusion_type
        self.rrf_k = int(rrf_k)
        self.alpha = float(alpha)
        self.filter_active = filter_active
        self.filter_threshold = 0.0 if fusion_type == "rrf" and filter_threshold is None else filter_threshold

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        self.dense_retriever = dense_retriever or FAISSRetriever(
            vectorstore=vectorstore,
            top_k=self.candidate_k,
            candidate_k=self.candidate_k,
            filter_active=filter_active,
            enable_multi_query=False,
        )
        self.sparse_retriever = sparse_retriever or SparseRetriever(
            documents=documents or [],
            top_k=self.candidate_k,
            candidate_k=self.candidate_k,
            filter_active=filter_active,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        if not query.strip():
            return []
        state = self.run({"query": query})
        results = state["retrieval_results"]
        if top_k is None:
            return results
        return results[:top_k]

    def _fuse_results(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        if self.fusion_type == "weighted":
            return weighted_fusion(dense_results=dense_results, sparse_results=sparse_results, alpha=self.alpha)
        if self.fusion_type == "rrf":
            return reciprocal_rank_fusion(dense_results=dense_results, sparse_results=sparse_results, k=self.rrf_k)
        raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        queries = self.get_queries_from_state(state)
        query = self.get_query_from_state(state)
        if not queries and query:
            queries = [query]

        if not queries:
            state["retrieval_results"] = []
            state["retrieval"] = self.build_metadata(
                strategy="hybrid",
                input_queries=[],
                num_queries=0,
                top_k=self.top_k,
                candidate_k=self.candidate_k,
                fusion_type=self.fusion_type,
                rrf_k=self.rrf_k,
                alpha=self.alpha,
                alpha_used=self.fusion_type == "weighted",
                dense_count=0,
                sparse_count=0,
                output_count=0,
                filter_active=self.filter_active,
                per_query_counts=[],
                filtered_inactive_count=0,
                sparse_fallback=False,
                sparse_error=None,
            )
            return state

        dense_results: List[RetrievalResult] = []
        sparse_results: List[RetrievalResult] = []
        per_query_counts: List[dict] = []
        sparse_fallback = False
        sparse_error = None

        for current_query in queries:
            dense_query_results = self.dense_retriever.retrieve(current_query, top_k=self.candidate_k)
            dense_results.extend(dense_query_results)

            sparse_query_results: List[RetrievalResult] = []
            try:
                sparse_query_results = self.sparse_retriever.retrieve(current_query, top_k=self.candidate_k)
            except Exception as exc:
                sparse_fallback = True
                sparse_error = str(exc)
                sparse_query_results = []

            sparse_results.extend(sparse_query_results)
            per_query_counts.append(
                {
                    "query": current_query,
                    "dense_count": len(dense_query_results),
                    "sparse_count": len(sparse_query_results),
                }
            )

        dense_count = len(dense_results)
        sparse_count = len(sparse_results)

        dense_results = deduplicate_results(dense_results)
        sparse_results = deduplicate_results(sparse_results)
        fused_results = self._fuse_results(dense_results=dense_results, sparse_results=sparse_results)
        fused_results = deduplicate_results(fused_results)

        filtered_inactive_count = 0
        if self.filter_active:
            fused_results, filtered_inactive_count = filter_active_results(fused_results)

        fused_results = filter_low_score_results(fused_results, threshold=self.filter_threshold)
        fused_results.sort(key=get_effective_score, reverse=True)
        fused_results = fused_results[: self.top_k]

        state["retrieval_results"] = fused_results
        state["retrieval"] = self.build_metadata(
            strategy="hybrid",
            input_queries=queries,
            num_queries=len(queries),
            top_k=self.top_k,
            candidate_k=self.candidate_k,
            fusion_type=self.fusion_type,
            rrf_k=self.rrf_k,
            alpha=self.alpha,
            alpha_used=self.fusion_type == "weighted",
            dense_count=dense_count,
            sparse_count=sparse_count,
            output_count=len(fused_results),
            filter_active=self.filter_active,
            per_query_counts=per_query_counts,
            filtered_inactive_count=filtered_inactive_count,
            sparse_fallback=sparse_fallback,
            sparse_error=sparse_error,
        )
        return state
