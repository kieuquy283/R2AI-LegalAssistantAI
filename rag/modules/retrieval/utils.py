from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterable, List, Sequence

from .schemas import RetrievalResult


STOPWORDS = {
    "và",
    "là",
    "của",
    "cho",
    "với",
    "the",
    "is",
    "are",
    "of",
    "to",
    "a",
    "an",
}


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_bm25(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    basic_tokens = [
        token
        for token in re.findall(r"\w+", normalized)
        if token not in STOPWORDS and len(token) > 1
    ]
    if not basic_tokens:
        return []

    bigrams = [
        f"{basic_tokens[index]}_{basic_tokens[index + 1]}"
        for index in range(len(basic_tokens) - 1)
    ]
    return basic_tokens + bigrams


def min_max_normalize(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(min_score, max_score):
        return [1.0 for _ in scores]

    denominator = max_score - min_score
    return [float(score - min_score) / float(denominator) for score in scores]


def dense_distance_to_similarity(distance: float, max_distance: float) -> float:
    return float(max_distance - distance)


def normalize_dense_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
    if not results:
        return results

    raw_scores = [result.raw_score if result.raw_score is not None else result.score for result in results]
    max_distance = max(raw_scores)
    converted = [dense_distance_to_similarity(score, max_distance) for score in raw_scores]
    normalized = min_max_normalize(converted)

    for result, similarity, score in zip(results, converted, normalized):
        result.normalized_score = float(score)
        result.score = float(score)
        result.dense_score = float(score)
        result.final_score = result.final_score if result.final_score is not None else float(score)
        if result.raw_score is None:
            result.raw_score = float(max_distance - similarity)

    return results


def normalize_sparse_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
    if not results:
        return results

    raw_scores = [result.raw_score if result.raw_score is not None else result.score for result in results]
    normalized = min_max_normalize(raw_scores)

    for result, score in zip(results, normalized):
        result.normalized_score = float(score)
        result.score = float(score)
        result.sparse_score = float(score)
        result.final_score = result.final_score if result.final_score is not None else float(score)
        if result.raw_score is None:
            result.raw_score = float(result.score)

    return results


def get_effective_score(result: RetrievalResult) -> float:
    if getattr(result, "final_score", None) is not None:
        return float(result.final_score)
    if getattr(result, "score", None) is not None:
        return float(result.score)
    return 0.0


def is_active_result(result: RetrievalResult) -> bool:
    metadata = result.metadata or {}
    return metadata.get("is_active", True) is not False


def _dedup_key(result: RetrievalResult) -> str:
    if result.chunk_id:
        return f"chunk_id:{result.chunk_id}"
    metadata = result.metadata or {}
    content_hash = metadata.get("content_hash")
    if content_hash:
        return f"content_hash:{content_hash}"
    text_hash = hashlib.md5(result.text.encode("utf-8")).hexdigest()
    return f"text_hash:{text_hash}"


def _merge_sources(existing: RetrievalResult, current: RetrievalResult) -> None:
    merged_sources = set(existing.sources or [existing.source])
    merged_sources.update(current.sources or [current.source])
    existing.sources = sorted(merged_sources)
    existing.metadata = dict(existing.metadata or {})
    existing.metadata["retrieval_sources"] = list(existing.sources)


def deduplicate_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
    unique_results = {}

    for result in results:
        key = _dedup_key(result)
        existing = unique_results.get(key)
        if existing is None:
            unique_results[key] = result
            continue

        if get_effective_score(result) > get_effective_score(existing):
            _merge_sources(result, existing)
            unique_results[key] = result
        else:
            _merge_sources(existing, result)

    deduplicated = sorted(unique_results.values(), key=get_effective_score, reverse=True)
    for rank, result in enumerate(deduplicated, start=1):
        result.retrieval_rank = rank
    return deduplicated


def filter_low_score_results(
    results: List[RetrievalResult],
    threshold: float | None = 0.1,
) -> List[RetrievalResult]:
    if threshold is None:
        return results
    return [result for result in results if get_effective_score(result) >= threshold]


def filter_active_results(results: List[RetrievalResult]) -> tuple[List[RetrievalResult], int]:
    filtered = [result for result in results if is_active_result(result)]
    return filtered, len(results) - len(filtered)


def summarize_results(results: List[RetrievalResult]) -> List[dict]:
    summary = []
    for result in results:
        summary.append(
            {
                "chunk_id": result.chunk_id,
                "score": round(result.score, 4),
                "final_score": None if result.final_score is None else round(result.final_score, 4),
                "source": result.source,
                "sources": list(result.sources or [result.source]),
                "preview": result.text[:120],
            }
        )
    return summary
