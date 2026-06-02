from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseReranker
from .schemas import RerankResult
from .utils import convert_to_rerank_result


class NoReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        retrieval_results,
    ) -> List[RerankResult]:
        results: List[RerankResult] = []
        for index, item in enumerate(retrieval_results or []):
            retrieval_result = convert_to_rerank_result(
                item,
                index,
                source="no_reranker",
            )
            retrieval_result.rerank_rank = retrieval_result.retrieval_rank
            retrieval_result.rank_delta = 0
            results.append(retrieval_result)
        return results

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        retrieval_results = self.get_retrieval_results_from_state(state)
        reranked_results = self.rerank(query=self.get_query_from_state(state), retrieval_results=retrieval_results)
        state["reranked_results"] = reranked_results
        state["reranking"] = self.build_metadata(
            strategy="no_reranker",
            rerank_applied=False,
            input_count=len(retrieval_results),
            output_count=len(reranked_results),
            model_name=None,
        )
        return state
