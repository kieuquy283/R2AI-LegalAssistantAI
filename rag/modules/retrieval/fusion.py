from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .schemas import RetrievalResult
from .utils import deduplicate_results, get_effective_score


def _as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dense_value(result: RetrievalResult) -> float:
    """
    Prefer the explicit dense score produced by dense_retriever.
    Fall back to normalized_score/score only when dense_score is missing.
    """
    if result.dense_score is not None:
        return _as_float(result.dense_score)
    if result.normalized_score is not None:
        return _as_float(result.normalized_score)
    return _as_float(result.score)


def _sparse_value(result: RetrievalResult) -> float:
    """
    Prefer explicit sparse_score for sparse/BM25 results.
    Fall back to normalized_score/score only when sparse_score is missing.
    """
    if result.sparse_score is not None:
        return _as_float(result.sparse_score)
    if result.normalized_score is not None:
        return _as_float(result.normalized_score)
    return _as_float(result.score)


def _copy_result(result: RetrievalResult) -> RetrievalResult:
    copied = RetrievalResult(
        chunk_id=result.chunk_id,
        text=result.text,
        score=result.score,
        source=result.source,
        metadata=dict(result.metadata or {}),
        retrieval_rank=result.retrieval_rank,
        rerank_score=result.rerank_score,
        final_score=result.final_score,
        raw_score=result.raw_score,
        normalized_score=result.normalized_score,
        dense_score=result.dense_score,
        sparse_score=result.sparse_score,
        sources=list(result.sources or [result.source]),
        retriever_name=result.retriever_name,
    )
    copied.metadata["retrieval_sources"] = list(copied.sources or [])
    return copied


def weighted_fusion(
    dense_results: List[RetrievalResult],
    sparse_results: List[RetrievalResult],
    alpha: float = 0.5,
) -> List[RetrievalResult]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    merged: Dict[str, RetrievalResult] = {}

    for result in dense_results:
        copied = _copy_result(result)

        dense_score = _dense_value(result)

        copied.dense_score = dense_score
        copied.sparse_score = _as_float(copied.sparse_score, 0.0)
        copied.final_score = alpha * dense_score
        copied.sources = ["dense"]
        copied.metadata["retrieval_sources"] = ["dense"]

        # Keep raw dense debug value in metadata if available.
        copied.metadata["dense_score_before_fusion"] = dense_score
        copied.metadata["raw_score_before_fusion"] = _as_float(result.raw_score, _as_float(result.score))

        merged[copied.chunk_id] = copied

    for result in sparse_results:
        chunk_id = result.chunk_id
        sparse_score = _sparse_value(result)

        if chunk_id in merged:
            merged_result = merged[chunk_id]

            # Preserve existing dense score from dense result.
            dense_score = _as_float(merged_result.dense_score, 0.0)

            merged_result.sparse_score = sparse_score
            merged_result.final_score = (alpha * dense_score) + ((1 - alpha) * sparse_score)
            merged_result.sources = sorted(set((merged_result.sources or []) + ["sparse"]))
            merged_result.metadata["retrieval_sources"] = list(merged_result.sources)
            merged_result.metadata["sparse_score_before_fusion"] = sparse_score
        else:
            copied = _copy_result(result)

            copied.dense_score = _as_float(copied.dense_score, 0.0)
            copied.sparse_score = sparse_score
            copied.final_score = (1 - alpha) * sparse_score
            copied.sources = ["sparse"]
            copied.metadata["retrieval_sources"] = ["sparse"]
            copied.metadata["sparse_score_before_fusion"] = sparse_score

            merged[chunk_id] = copied

    final_results = sorted(merged.values(), key=get_effective_score, reverse=True)
    for rank, result in enumerate(final_results, start=1):
        result.retrieval_rank = rank
    return deduplicate_results(final_results)


def reciprocal_rank_fusion(
    dense_results: List[RetrievalResult],
    sparse_results: List[RetrievalResult],
    k: int = 60,
) -> List[RetrievalResult]:
    scores = defaultdict(float)
    merged_objects: Dict[str, RetrievalResult] = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.chunk_id
        scores[chunk_id] += 1.0 / (k + rank)

        dense_score = _dense_value(result)

        if chunk_id not in merged_objects:
            copied = _copy_result(result)
            copied.sources = ["dense"]
            copied.metadata["retrieval_sources"] = ["dense"]
            merged_objects[chunk_id] = copied

        merged_objects[chunk_id].dense_score = dense_score
        merged_objects[chunk_id].metadata["dense_score_before_fusion"] = dense_score
        merged_objects[chunk_id].metadata["raw_score_before_fusion"] = _as_float(
            result.raw_score,
            _as_float(result.score),
        )

    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result.chunk_id
        scores[chunk_id] += 1.0 / (k + rank)

        sparse_score = _sparse_value(result)

        if chunk_id not in merged_objects:
            copied = _copy_result(result)
            copied.sources = ["sparse"]
            copied.metadata["retrieval_sources"] = ["sparse"]
            merged_objects[chunk_id] = copied
        else:
            sources = sorted(set((merged_objects[chunk_id].sources or []) + ["sparse"]))
            merged_objects[chunk_id].sources = sources
            merged_objects[chunk_id].metadata["retrieval_sources"] = sources

        # Do not touch dense_score here. Sparse result must not overwrite dense score.
        merged_objects[chunk_id].sparse_score = sparse_score
        merged_objects[chunk_id].metadata["sparse_score_before_fusion"] = sparse_score

    final_results: List[RetrievalResult] = []
    for chunk_id, score in scores.items():
        item = merged_objects[chunk_id]
        item.final_score = float(score)
        final_results.append(item)

    final_results.sort(key=get_effective_score, reverse=True)
    for rank, result in enumerate(final_results, start=1):
        result.retrieval_rank = rank
    return deduplicate_results(final_results)