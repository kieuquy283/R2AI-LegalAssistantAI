from __future__ import annotations

from typing import Any, Iterable, List

from .schemas import RetrievalConfidence


def _extract_cid(item: Any) -> str:
    metadata = dict(getattr(item, "metadata", {}) or {})
    chunk_id = str(getattr(item, "chunk_id", "") or "").strip()
    return str(metadata.get("cid") or metadata.get("chunk_id") or chunk_id).strip()


def _extract_text(item: Any) -> str:
    text = getattr(item, "text", None)
    if text is None:
        text = getattr(item, "page_content", "")
    return str(text or "").strip()


def _extract_ranked_score(item: Any) -> float | None:
    for field_name in (
        "final_score",
        "rerank_score",
        "normalized_score",
        "score",
        "raw_score",
        "retrieval_score",
    ):
        value = getattr(item, field_name, None)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _compute_gap(scores: List[float]) -> float | None:
    if len(scores) < 2:
        return None
    first = abs(scores[0])
    second = abs(scores[1])
    denominator = max(first + second, 1e-8)
    return abs(scores[0] - scores[1]) / denominator


def compute_retrieval_confidence(
    results: Iterable[Any],
    *,
    expected_top_k: int = 5,
) -> RetrievalConfidence:
    items = list(results or [])
    doc_count = len(items)
    cids = [_extract_cid(item) for item in items]
    unique_cids = [cid for cid in dict.fromkeys(cids) if cid]
    unique_cid_count = len(unique_cids)
    has_cid_metadata = all(bool(cid) for cid in cids) if items else False

    non_empty_count = sum(1 for item in items if _extract_text(item))
    non_empty_content_ratio = float(non_empty_count / doc_count) if doc_count else 0.0
    duplicate_rate = 0.0
    if doc_count:
        duplicate_rate = max(float(doc_count - unique_cid_count) / float(doc_count), 0.0)

    raw_scores = [_extract_ranked_score(item) for item in items]
    scores = [score for score in raw_scores if score is not None]
    score_signal_available = len(scores) > 0
    top_score = None
    if scores:
        first_score = scores[0]
        if 0.0 <= first_score <= 1.0:
            top_score = first_score
    score_gap = _compute_gap(scores)

    count_signal = min(doc_count / max(expected_top_k, 1), 1.0)
    unique_signal = min(unique_cid_count / max(expected_top_k, 1), 1.0)
    cid_signal = 1.0 if has_cid_metadata else 0.4
    duplicate_signal = max(0.0, 1.0 - duplicate_rate)
    top_score_signal = top_score if top_score is not None else 0.5
    gap_signal = score_gap if score_gap is not None else 0.5

    score = (
        0.25 * count_signal
        + 0.25 * unique_signal
        + 0.15 * non_empty_content_ratio
        + 0.10 * cid_signal
        + 0.10 * duplicate_signal
        + 0.075 * top_score_signal
        + 0.075 * gap_signal
    )
    score = max(0.0, min(float(score), 1.0))

    reasons: List[str] = []
    if doc_count == 0:
        reasons.append("no_documents")
    if doc_count and doc_count < expected_top_k:
        reasons.append("too_few_documents")
    if unique_cid_count <= 1 and doc_count > 1:
        reasons.append("low_cid_diversity")
    if duplicate_rate >= 0.4:
        reasons.append("high_duplicate_rate")
    if non_empty_content_ratio < 1.0:
        reasons.append("empty_content_present")
    if not has_cid_metadata and doc_count > 0:
        reasons.append("missing_cid_metadata")
    if not score_signal_available:
        reasons.append("missing_scores")

    return RetrievalConfidence(
        score=score,
        doc_count=doc_count,
        unique_cid_count=unique_cid_count,
        has_cid_metadata=has_cid_metadata,
        non_empty_content_ratio=non_empty_content_ratio,
        duplicate_rate=duplicate_rate,
        score_signal_available=score_signal_available,
        top_score=top_score,
        score_gap=score_gap,
        reasons=reasons,
    )

