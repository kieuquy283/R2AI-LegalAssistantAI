from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .schemas import RerankResult


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        retrieval_results,
    ) -> List[RerankResult]:
        raise NotImplementedError

    def get_query_from_state(self, state: Dict[str, Any]) -> str:
        return (
            state.get("rewritten_query")
            or state.get("query")
            or state.get("question")
            or ""
        ).strip()

    def get_retrieval_results_from_state(self, state: Dict[str, Any]):
        return (
            state.get("retrieval_results")
            or state.get("retrieved_docs")
            or state.get("documents")
            or []
        )

    def build_metadata(
        self,
        *,
        strategy: str,
        rerank_applied: bool,
        input_count: int,
        output_count: int,
        model_name: str | None,
        **extra: Any,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "strategy": strategy,
            "rerank_applied": rerank_applied,
            "input_count": int(input_count),
            "output_count": int(output_count),
            "model_name": model_name,
        }
        metadata.update(extra)
        return metadata

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self.get_query_from_state(state)
        retrieval_results = self.get_retrieval_results_from_state(state)
        reranked_results = self.rerank(query=query, retrieval_results=retrieval_results)
        state["reranked_results"] = reranked_results
        if "reranking" not in state:
            state["reranking"] = self.build_metadata(
                strategy=self.__class__.__name__.lower(),
                rerank_applied=True,
                input_count=len(retrieval_results),
                output_count=len(reranked_results),
                model_name=getattr(self, "model_name", None),
            )
        return state
