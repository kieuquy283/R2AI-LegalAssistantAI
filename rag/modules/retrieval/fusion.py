from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .schemas import RetrievalResult
from .utils import deduplicate_results, get_effective_score


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
        copied.dense_score = result.score
        copied.sparse_score = copied.sparse_score
        copied.final_score = alpha * result.score
        copied.sources = ["dense"]
        copied.metadata["retrieval_sources"] = ["dense"]
        merged[copied.chunk_id] = copied

    for result in sparse_results:
        chunk_id = result.chunk_id
        sparse_score = result.score
        if chunk_id in merged:
            merged_result = merged[chunk_id]
            merged_result.sparse_score = sparse_score
            merged_result.final_score = (alpha * (merged_result.dense_score or 0.0)) + (
                (1 - alpha) * sparse_score
            )
            merged_result.sources = sorted(set((merged_result.sources or []) + ["sparse"]))
            merged_result.metadata["retrieval_sources"] = list(merged_result.sources)
        else:
            copied = _copy_result(result)
            copied.dense_score = copied.dense_score
            copied.sparse_score = sparse_score
            copied.final_score = (1 - alpha) * sparse_score
            copied.sources = ["sparse"]
            copied.metadata["retrieval_sources"] = ["sparse"]
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
        if chunk_id not in merged_objects:
            copied = _copy_result(result)
            copied.sources = ["dense"]
            copied.metadata["retrieval_sources"] = ["dense"]
            merged_objects[chunk_id] = copied
        merged_objects[chunk_id].dense_score = result.score

    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result.chunk_id
        scores[chunk_id] += 1.0 / (k + rank)
        if chunk_id not in merged_objects:
            copied = _copy_result(result)
            copied.sources = ["sparse"]
            copied.metadata["retrieval_sources"] = ["sparse"]
            merged_objects[chunk_id] = copied
        else:
            sources = sorted(set((merged_objects[chunk_id].sources or []) + ["sparse"]))
            merged_objects[chunk_id].sources = sources
            merged_objects[chunk_id].metadata["retrieval_sources"] = sources
        merged_objects[chunk_id].sparse_score = result.score

    final_results: List[RetrievalResult] = []
    for chunk_id, score in scores.items():
        item = merged_objects[chunk_id]
        item.final_score = float(score)
        final_results.append(item)

    final_results.sort(key=get_effective_score, reverse=True)
    for rank, result in enumerate(final_results, start=1):
        result.retrieval_rank = rank
    return deduplicate_results(final_results)
