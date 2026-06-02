from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .schemas import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        raise NotImplementedError

    def get_query_from_state(self, state: Dict[str, Any]) -> str:
        return (
            state.get("rewritten_query")
            or state.get("query")
            or state.get("question")
            or ""
        ).strip()

    def get_queries_from_state(self, state: Dict[str, Any]) -> List[str]:
        queries = state.get("queries")
        if isinstance(queries, list):
            normalized_queries = [str(query).strip() for query in queries if str(query).strip()]
            if normalized_queries:
                return normalized_queries

        query = self.get_query_from_state(state)
        return [query] if query else []

    def build_metadata(self, **kwargs: Any) -> Dict[str, Any]:
        return dict(kwargs)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self.get_query_from_state(state)
        results = self.retrieve(query=query, top_k=getattr(self, "top_k", None))
        state["retrieval_results"] = results
        if "retrieval" not in state:
            state["retrieval"] = self.build_metadata(
                strategy=self.__class__.__name__.lower(),
                input_queries=[query] if query else [],
                top_k=getattr(self, "top_k", None),
                candidate_k=getattr(self, "candidate_k", None),
                output_count=len(results),
                filter_active=getattr(self, "filter_active", False),
            )
        return state
