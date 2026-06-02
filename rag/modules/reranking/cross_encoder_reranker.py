from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .base import BaseReranker
from .schemas import RerankResult
from .utils import (
    combine_scores,
    convert_to_rerank_result,
    deduplicate_reranked_results,
    normalize_scores,
    summarize_rerank_results,
)


class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        candidate_top_k: int = 30,
        batch_size: int = 8,
        max_length: int = 512,
        score_alpha: float = 0.2,
        normalization: str = "sigmoid",
        top_k: int | None = None,
        output_top_k: int | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.candidate_top_k = int(candidate_top_k)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.score_alpha = float(score_alpha)
        self.normalization = normalization
        self.top_k = top_k
        self.output_top_k = output_top_k

        if self.candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be > 0")
        if not 0.0 <= self.score_alpha <= 1.0:
            raise ValueError("score_alpha must be between 0 and 1")

        self.model = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError(
                "Failed to import sentence_transformers.CrossEncoder for reranking."
            ) from exc

        try:
            return CrossEncoder(self.model_name, max_length=self.max_length)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cross-encoder reranker model '{self.model_name}'."
            ) from exc

    def _normalize_scores(self, raw_scores: Sequence[float]) -> List[float]:
        return normalize_scores(raw_scores, strategy=self.normalization)

    def _build_pairs(
        self,
        query: str,
        retrieval_results,
    ) -> tuple[List[tuple[str, str]], List[RerankResult]]:
        pairs = []
        converted_results: List[RerankResult] = []
        for index, item in enumerate(retrieval_results):
            converted = convert_to_rerank_result(item, index, source="retrieval")
            converted_results.append(converted)
            pairs.append((query, converted.text))
        return pairs, converted_results

    def rerank(
        self,
        query: str,
        retrieval_results,
    ) -> List[RerankResult]:
        if not retrieval_results:
            return []

        if not query.strip():
            return [
                convert_to_rerank_result(item, index, source="no_reranker")
                for index, item in enumerate(retrieval_results)
            ]

        candidate_items = list(retrieval_results[: self.candidate_top_k])
        remaining_items = list(retrieval_results[self.candidate_top_k :])
        pairs, candidate_results = self._build_pairs(query, candidate_items)

        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        normalized_scores = self._normalize_scores(raw_scores)
        reranked_results: List[RerankResult] = []

        for converted, raw_score, normalized_score in zip(
            candidate_results, raw_scores, normalized_scores
        ):
            converted.rerank_score = float(normalized_score)
            converted.final_score = combine_scores(
                retrieval_score=converted.retrieval_score,
                rerank_score=converted.rerank_score,
                alpha=self.score_alpha,
            )
            converted.source = "reranker"
            converted.reranker_name = self.model_name
            converted.raw_rerank_score = float(raw_score)
            converted.normalized_rerank_score = float(normalized_score)
            reranked_results.append(converted)

        reranked_results.sort(key=lambda item: item.final_score, reverse=True)
        for rerank_rank, result in enumerate(reranked_results, start=1):
            result.rerank_rank = rerank_rank
            result.rank_delta = result.retrieval_rank - result.rerank_rank

        reranked_results = deduplicate_reranked_results(reranked_results)

        for original_index, item in enumerate(remaining_items, start=len(candidate_items)):
            passthrough = convert_to_rerank_result(item, original_index, source="no_reranker")
            passthrough.rerank_rank = len(reranked_results) + 1
            passthrough.rank_delta = passthrough.retrieval_rank - passthrough.rerank_rank
            reranked_results.append(passthrough)

        if self.output_top_k is not None:
            reranked_results = reranked_results[: self.output_top_k]
        elif self.top_k is not None:
            reranked_results = reranked_results[: self.top_k]

        return reranked_results

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self.get_query_from_state(state)
        retrieval_results = self.get_retrieval_results_from_state(state)

        fallback_reason = None
        if not query.strip():
            fallback_reason = "empty_query"
            reranked_results = [
                convert_to_rerank_result(item, index, source="no_reranker")
                for index, item in enumerate(retrieval_results or [])
            ]
            rerank_applied = False
        else:
            try:
                reranked_results = self.rerank(query=query, retrieval_results=retrieval_results)
                rerank_applied = bool(retrieval_results)
            except Exception:
                fallback_reason = "prediction_error"
                reranked_results = [
                    convert_to_rerank_result(item, index, source="no_reranker")
                    for index, item in enumerate(retrieval_results or [])
                ]
                rerank_applied = False

        state["reranked_results"] = reranked_results
        state["reranking"] = self.build_metadata(
            strategy="reranker",
            rerank_applied=rerank_applied,
            input_count=len(retrieval_results),
            output_count=len(reranked_results),
            model_name=self.model_name,
            candidate_top_k=self.candidate_top_k,
            score_alpha=self.score_alpha,
            normalization=self.normalization,
            batch_size=self.batch_size,
            max_length=self.max_length,
            fallback_reason=fallback_reason,
            top_results=summarize_rerank_results(reranked_results[:5]),
        )
        return state


class Reranker(CrossEncoderReranker):
    pass
