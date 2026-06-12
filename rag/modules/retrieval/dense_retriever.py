from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseRetriever
from .schemas import RetrievalResult
from .utils import (
    deduplicate_results,
    filter_active_results,
    get_effective_score,
    normalize_dense_scores,
)


class FAISSRetriever(BaseRetriever):
    """
    Dense retriever using FAISS similarity search.
    """

    def __init__(
        self,
        vectorstore,
        top_k: int = 5,
        candidate_k: Optional[int] = None,
        filter_active: bool = True,
        enable_multi_query: bool = False,
    ) -> None:
        self.vectorstore = vectorstore
        self.top_k = int(top_k)
        self.candidate_k = int(candidate_k or max(self.top_k * 2, self.top_k + 5))
        self.filter_active = filter_active
        self.enable_multi_query = enable_multi_query

    def _dense_score_mode(self) -> str:
        """
        LangChain FAISS often returns distance-like scores where lower is better.
        LangChain/Qdrant with cosine returns similarity-like scores where higher is better.
        """
        vectorstore_type = type(self.vectorstore).__name__.lower()
        vectorstore_module = type(self.vectorstore).__module__.lower()
        combined = f"{vectorstore_module}.{vectorstore_type}"

        if "qdrant" in combined:
            return "similarity"

        if "faiss" in combined:
            return "distance"

        # Safe default for most cosine vectorstores.
        return "similarity"

    def _search(self, query: str, candidate_k: int) -> List[RetrievalResult]:
        docs_scores = self.vectorstore.similarity_search_with_score(query=query, k=candidate_k)
        results: List[RetrievalResult] = []

        for rank, (doc, raw_score) in enumerate(docs_scores, start=1):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            text = str(getattr(doc, "page_content", "") or "").strip()
            if not text:
                continue

            chunk_id = str(metadata.get("chunk_id", rank))
            raw_score = float(raw_score)

            result = RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                score=raw_score,
                source="dense",
                metadata=metadata,
                retrieval_rank=rank,
                raw_score=raw_score,
                dense_score=raw_score,
                retriever_name=type(self.vectorstore).__name__,
                sources=["dense"],
            )
            result.metadata["retrieval_sources"] = ["dense"]
            result.metadata["raw_dense_score"] = raw_score
            result.metadata["vectorstore_type"] = type(self.vectorstore).__name__
            result.metadata["vectorstore_module"] = type(self.vectorstore).__module__
            results.append(result)

        return normalize_dense_scores(results, score_mode=self._dense_score_mode())

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        if not query.strip():
            return []

        result_top_k = int(top_k or self.top_k)
        results = self._search(query=query, candidate_k=self.candidate_k)
        if self.filter_active:
            results, _ = filter_active_results(results)
        results = deduplicate_results(results)
        results.sort(key=get_effective_score, reverse=True)
        return results[:result_top_k]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        queries = self.get_queries_from_state(state)
        query = queries[0] if queries else ""

        if not query:
            state["retrieval_results"] = []
            state["retrieval"] = self.build_metadata(
                strategy="faiss",
                input_queries=[],
                top_k=self.top_k,
                candidate_k=self.candidate_k,
                output_count=0,
                filter_active=self.filter_active,
                filtered_inactive_count=0,
            )
            return state

        results = self._search(query=query, candidate_k=self.candidate_k)
        filtered_inactive_count = 0
        if self.filter_active:
            results, filtered_inactive_count = filter_active_results(results)
        results = deduplicate_results(results)
        results.sort(key=get_effective_score, reverse=True)
        results = results[: self.top_k]

        state["retrieval_results"] = results
        state["retrieval"] = self.build_metadata(
            strategy="faiss",
            input_queries=[query],
            top_k=self.top_k,
            candidate_k=self.candidate_k,
            output_count=len(results),
            filter_active=self.filter_active,
            filtered_inactive_count=filtered_inactive_count,
        )
        return state


class DenseRetriever(FAISSRetriever):
    pass
