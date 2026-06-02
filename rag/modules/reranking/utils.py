from __future__ import annotations

import math
from typing import Any, Iterable, List, Sequence

from .schemas import RerankResult


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 1.0 if x > 0 else 0.0


def normalize_scores(
    scores: Sequence[float],
    strategy: str = "sigmoid",
) -> List[float]:
    if not scores:
        return []

    if strategy == "sigmoid":
        return [sigmoid(float(score)) for score in scores]

    if strategy == "minmax":
        min_score = min(scores)
        max_score = max(scores)
        if math.isclose(min_score, max_score):
            return [0.5 for _ in scores]
        denominator = max_score - min_score
        return [float(score - min_score) / float(denominator) for score in scores]

    if strategy == "none":
        return [float(score) for score in scores]

    raise ValueError(f"Unsupported normalization strategy: {strategy}")


def normalize_rerank_scores(scores: Sequence[float]) -> List[float]:
    return normalize_scores(scores, strategy="sigmoid")


def combine_scores(
    retrieval_score: float,
    rerank_score: float,
    alpha: float = 0.2,
) -> float:
    return (alpha * retrieval_score) + ((1 - alpha) * rerank_score)


def _safe_metadata(item: Any) -> dict:
    metadata = getattr(item, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def extract_text(item: Any) -> str:
    if hasattr(item, "text"):
        return str(getattr(item, "text") or "").strip()
    if hasattr(item, "page_content"):
        return str(getattr(item, "page_content") or "").strip()
    return str(getattr(item, "content", "") or "").strip()


def extract_chunk_id(item: Any, index: int) -> str:
    if hasattr(item, "chunk_id") and getattr(item, "chunk_id") is not None:
        return str(getattr(item, "chunk_id"))

    metadata = _safe_metadata(item)
    if metadata.get("chunk_id") is not None:
        return str(metadata["chunk_id"])

    return f"chunk_{index}"


def extract_retrieval_score(item: Any) -> float:
    for attribute in ("final_score", "score", "retrieval_score"):
        value = getattr(item, attribute, None)
        if value is not None:
            return float(value)

    metadata = _safe_metadata(item)
    for key in ("score", "final_score", "retrieval_score"):
        value = metadata.get(key)
        if value is not None:
            return float(value)

    return 0.0


def extract_retrieval_rank(item: Any, index: int) -> int:
    value = getattr(item, "retrieval_rank", None)
    if value is None or int(value) <= 0:
        return index + 1
    return int(value)


def convert_to_rerank_result(
    item: Any,
    index: int,
    *,
    rerank_score: float | None = None,
    final_score: float | None = None,
    source: str = "reranker",
    reranker_name: str | None = None,
    raw_rerank_score: float | None = None,
    normalized_rerank_score: float | None = None,
) -> RerankResult:
    retrieval_score = extract_retrieval_score(item)
    retrieval_rank = extract_retrieval_rank(item, index)
    result = RerankResult(
        chunk_id=extract_chunk_id(item, index),
        text=extract_text(item),
        retrieval_score=retrieval_score,
        rerank_score=retrieval_score if rerank_score is None else float(rerank_score),
        final_score=retrieval_score if final_score is None else float(final_score),
        metadata=_safe_metadata(item),
        retrieval_rank=retrieval_rank,
        rerank_rank=retrieval_rank,
        source=source,
        raw_rerank_score=raw_rerank_score,
        normalized_rerank_score=normalized_rerank_score,
        reranker_name=reranker_name,
    )
    if result.rerank_rank > 0:
        result.rank_delta = result.retrieval_rank - result.rerank_rank
    return result


def adaptive_threshold(
    rerank_scores: List[float],
    base_threshold: float = 0.5,
) -> float:
    if not rerank_scores:
        return base_threshold

    top_score = max(rerank_scores)
    if top_score >= 0.9:
        return 0.75
    if top_score >= 0.8:
        return 0.65
    if top_score >= 0.7:
        return 0.55
    return base_threshold


def select_top_contexts(
    reranked_results: List[RerankResult],
    min_contexts: int = 2,
    max_contexts: int = 8,
    relative_threshold: float = 0.8,
) -> List[RerankResult]:
    if not reranked_results:
        return []

    rerank_scores = [result.rerank_score for result in reranked_results]
    threshold = adaptive_threshold(rerank_scores)
    top_score = reranked_results[0].rerank_score
    selected: List[RerankResult] = []

    for result in reranked_results:
        if result.rerank_score < threshold:
            continue
        relative_score = result.rerank_score / max(top_score, 1e-8)
        if relative_score < relative_threshold:
            continue
        selected.append(result)
        if len(selected) >= max_contexts:
            break

    if len(selected) < min_contexts:
        return reranked_results[:min_contexts]

    return selected


def deduplicate_reranked_results(results: List[RerankResult]) -> List[RerankResult]:
    unique = {}

    for result in results:
        chunk_id = result.chunk_id
        existing = unique.get(chunk_id)
        if existing is None or result.final_score > existing.final_score:
            unique[chunk_id] = result

    deduplicated = sorted(unique.values(), key=lambda item: item.final_score, reverse=True)
    for rank, result in enumerate(deduplicated, start=1):
        result.rerank_rank = rank
        result.rank_delta = result.retrieval_rank - result.rerank_rank
    return deduplicated


def summarize_rerank_results(results: List[RerankResult]) -> List[dict]:
    summary = []
    for result in results:
        summary.append(
            {
                "chunk_id": result.chunk_id,
                "retrieval_rank": result.retrieval_rank,
                "rerank_rank": result.rerank_rank,
                "retrieval_score": round(result.retrieval_score, 4),
                "rerank_score": round(result.rerank_score, 4),
                "final_score": round(result.final_score, 4),
            }
        )
    return summary
