from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Iterable, Sequence

from rag.config.runtime import RetrievalRuntimeConfig, get_retrieval_runtime_config
from rag.modules.retrieval.utils import tokenize_for_bm25

_RRF_K = int(os.getenv("R2AI_RRF_K", "60"))


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d").replace("Đ", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _combined_text(candidate: dict[str, Any]) -> str:
    parts = [
        candidate.get("doc_number"),
        candidate.get("doc_title"),
        candidate.get("article"),
        candidate.get("clause"),
        candidate.get("citation"),
        candidate.get("content"),
        candidate.get("domain"),
    ]
    return "\n".join(str(part or "").strip() for part in parts if str(part or "").strip())


def _normalize_id(cid: str) -> str:
    """Strip level prefix (article:/chunk:/doc:) for cross-source dedup."""
    if ":" in cid:
        return cid.split(":", 1)[1]
    return cid


def _merge_candidates(candidates: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        raw_key = str(candidate.get("candidate_id") or "")
        if not raw_key:
            continue
        key = _normalize_id(raw_key)
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(candidate)
            continue
        # Merge scores: take max from either source
        existing["dense_score"] = max(float(existing.get("dense_score") or 0.0), float(candidate.get("dense_score") or 0.0))
        existing["bm25_score"] = max(float(existing.get("bm25_score") or 0.0), float(candidate.get("bm25_score") or 0.0))
        existing["exact_score"] = max(float(existing.get("exact_score") or 0.0), float(candidate.get("exact_score") or 0.0))
        existing["retrieval_method"] = "hybrid"
        for field in (
            "doc_id",
            "article_id",
            "chunk_id",
            "chunk_ref_id",
            "doc_number",
            "doc_title",
            "article",
            "clause",
            "citation",
            "domain",
            "source_url",
            "content",
            "source_dataset",
            "priority",
            "retrieval_level",
            "metadata",
        ):
            if not existing.get(field) and candidate.get(field):
                existing[field] = candidate.get(field)
    return merged


def _rrf_score(rank: int | None, k: int = _RRF_K) -> float:
    """Reciprocal rank: 1/(k + rank) for ranked items, 0 for not found."""
    if rank is None or rank < 0:
        return 0.0
    return 1.0 / (k + rank)


def rrf_fuse_candidates(
    query: str,
    *,
    dense_candidates: Sequence[dict[str, Any]],
    bm25_candidates: Sequence[dict[str, Any]],
    exact_candidates: Sequence[dict[str, Any]],
    preferred_domains: Sequence[str] | None = None,
    config: RetrievalRuntimeConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF). Combines rankings from dense, BM25, and exact retrievers.
    score = Σ 1/(k + rank_i) for each source where candidate appears.
    """
    config = config or get_retrieval_runtime_config()
    merged = _merge_candidates([*dense_candidates, *bm25_candidates, *exact_candidates])

    # Build rank lookups per source (normalized IDs for cross-source matching)
    dense_ranks: dict[str, int] = {}
    for i, c in enumerate(dense_candidates):
        key = _normalize_id(str(c.get("candidate_id") or ""))
        if key:
            dense_ranks[key] = i

    bm25_ranks: dict[str, int] = {}
    for i, c in enumerate(bm25_candidates):
        key = _normalize_id(str(c.get("candidate_id") or ""))
        if key:
            bm25_ranks[key] = i

    exact_ranks: dict[str, int] = {}
    for i, c in enumerate(exact_candidates):
        key = _normalize_id(str(c.get("candidate_id") or ""))
        if key:
            exact_ranks[key] = i

    query_tokens = set(tokenize_for_bm25(_normalize_text(query)))
    preferred = {str(item) for item in preferred_domains or [] if str(item)}
    query_normalized = _normalize_text(query)

    # Temporal relevance: extract year from doc_number (e.g., "45/2026/NĐ-CP" → 2026)
    _YEAR_RE = re.compile(r'(?:^|/)(\d{4})(?:/|$)')
    def _doc_year(candidate: dict) -> int | None:
        dn = str(candidate.get("doc_number") or "")
        m = _YEAR_RE.search(dn)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return None
    # Extract years mentioned in query (e.g., "2024", "năm 2024")
    _QUERY_YEAR_RE = re.compile(r'(?:năm\s+)?(\d{4})')
    query_years = {int(m.group(1)) for m in _QUERY_YEAR_RE.finditer(query) if 1900 <= int(m.group(1)) <= 2100}
    import datetime
    CURRENT_YEAR = datetime.datetime.now().year

    reranked: list[dict[str, Any]] = []
    bonus_cache: list[dict[str, float]] = []
    for candidate in merged.values():
        key = str(candidate.get("candidate_id") or "")
        dr = dense_ranks.get(key)
        br = bm25_ranks.get(key)
        er = exact_ranks.get(key)

        rrf_dense = _rrf_score(dr)
        rrf_bm25 = _rrf_score(br)
        rrf_exact = _rrf_score(er)

        source_count = sum(1 for r in [dr, br, er] if r is not None)
        rrf_sum = rrf_dense + rrf_bm25 + rrf_exact

        combined = _combined_text(candidate)
        normalized_combined = _normalize_text(combined)
        title_space = _normalize_text(" ".join([
            str(candidate.get("doc_title") or ""),
            str(candidate.get("citation") or ""),
            str(candidate.get("article") or ""),
        ]))
        text_tokens = set(tokenize_for_bm25(normalized_combined))
        title_tokens = set(tokenize_for_bm25(title_space))
        lexical_overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
        title_overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
        domain = str(candidate.get("domain") or "")
        domain_match = 1.0 if preferred and domain in preferred else 0.0
        wrong_domain_penalty = 0.2 if preferred and domain and domain not in preferred and domain != "business_law" else 0.0
        citation_match_val = 1.0 if str(candidate.get("article") or "").strip() and _normalize_text(str(candidate.get("article") or "")) in query_normalized else 0.0
        hf_priority_boost = 0.05 if str(candidate.get("source_dataset") or "") == "th1nhng0/vietnamese-legal-documents" and int(candidate.get("priority") or 0) == 1 else 0.0

        doc_year = _doc_year(candidate)
        temporal_boost = 0.0
        if doc_year is not None:
            if query_years:
                best_dist = min(abs(doc_year - qy) for qy in query_years)
                if best_dist == 0:
                    temporal_boost = 0.03
                elif best_dist == 1:
                    temporal_boost = 0.02
            else:
                years_since = CURRENT_YEAR - doc_year
                if years_since <= 2:
                    temporal_boost = 0.03
                elif years_since <= 5:
                    temporal_boost = 0.02
                elif years_since <= 10:
                    temporal_boost = 0.01

        candidate["dense_rank"] = dr
        candidate["bm25_rank"] = br
        candidate["exact_rank"] = er
        candidate["rrf_dense"] = round(rrf_dense, 6)
        candidate["rrf_bm25"] = round(rrf_bm25, 6)
        candidate["rrf_exact"] = round(rrf_exact, 6)
        candidate["rrf_sum"] = round(rrf_sum, 6)
        candidate["rrf_source_count"] = source_count
        candidate["title_overlap"] = round(title_overlap, 4)
        candidate["lexical_overlap"] = round(lexical_overlap, 4)
        candidate["domain_match"] = round(domain_match, 4)
        candidate["domain_score"] = round(domain_match, 4)
        candidate["citation_match"] = round(citation_match_val, 4)
        candidate["hf_priority_boost"] = round(hf_priority_boost, 4)
        candidate["wrong_domain_penalty"] = round(wrong_domain_penalty, 4)
        candidate["doc_year"] = doc_year
        candidate["temporal_boost"] = round(temporal_boost, 4)
        reranked.append(candidate)
        bonus_cache.append(dict(
            lexical_overlap=lexical_overlap,
            domain_match=domain_match,
            citation_match=citation_match_val,
            hf_priority_boost=hf_priority_boost,
            temporal_boost=temporal_boost,
            wrong_domain_penalty=wrong_domain_penalty,
        ))

    # Min-max normalize RRF sum across all candidates, then add bonuses
    raw_rrf_scores = [float(c["rrf_sum"]) for c in reranked]
    min_rrf, max_rrf = min(raw_rrf_scores), max(raw_rrf_scores)
    rrf_range = max(1e-8, max_rrf - min_rrf)

    for candidate, b in zip(reranked, bonus_cache):
        normalized_rrf = (candidate["rrf_sum"] - min_rrf) / rrf_range
        final_score = (
            normalized_rrf
            + b["lexical_overlap"] * 0.05
            + b["domain_match"] * 0.04
            + b["citation_match"] * 0.02
            + b["hf_priority_boost"]
            + b["temporal_boost"]
            - b["wrong_domain_penalty"]
        )
        candidate["final_score"] = round(max(0.0, final_score), 6)
        candidate["confidence"] = candidate["final_score"]
        candidate["metadata"] = {
            **dict(candidate.get("metadata") or {}),
            "doc_id": candidate.get("doc_id"),
            "doc_number": candidate.get("doc_number"),
            "doc_title": candidate.get("doc_title"),
            "article": candidate.get("article"),
            "clause": candidate.get("clause"),
            "citation": candidate.get("citation"),
            "domain": candidate.get("domain"),
            "source_url": candidate.get("source_url"),
        }

    reranked.sort(key=lambda item: float(item.get("final_score") or 0.0), reverse=True)

    # Global min-max normalization across fused results
    _min_max_normalize_scores(reranked)

    return reranked[: config.rerank_top_n]


def fuse_candidates(
    query: str,
    *,
    dense_candidates: Sequence[dict[str, Any]],
    bm25_candidates: Sequence[dict[str, Any]],
    exact_candidates: Sequence[dict[str, Any]],
    preferred_domains: Sequence[str] | None = None,
    config: RetrievalRuntimeConfig | None = None,
) -> list[dict[str, Any]]:
    use_rrf = os.getenv("R2AI_USE_RRF", "").strip().lower() in {"1", "true", "yes"}
    if use_rrf:
        return rrf_fuse_candidates(
            query,
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
            exact_candidates=exact_candidates,
            preferred_domains=preferred_domains,
            config=config,
        )

    config = config or get_retrieval_runtime_config()
    merged = _merge_candidates([*dense_candidates, *bm25_candidates, *exact_candidates])
    query_tokens = set(tokenize_for_bm25(_normalize_text(query)))
    preferred = {str(item) for item in preferred_domains or [] if str(item)}
    query_normalized = _normalize_text(query)
    reranked: list[dict[str, Any]] = []
    for candidate in merged.values():
        combined = _combined_text(candidate)
        normalized_combined = _normalize_text(combined)
        title_space = _normalize_text(" ".join(
            [
                str(candidate.get("doc_title") or ""),
                str(candidate.get("citation") or ""),
                str(candidate.get("article") or ""),
            ]
        ))
        text_tokens = set(tokenize_for_bm25(normalized_combined))
        title_tokens = set(tokenize_for_bm25(title_space))
        lexical_overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
        title_overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
        domain = str(candidate.get("domain") or "")
        domain_match = 1.0 if preferred and domain in preferred else 0.0
        wrong_domain_penalty = 0.2 if preferred and domain and domain not in preferred and domain != "business_law" else 0.0
        citation_match = 1.0 if str(candidate.get("article") or "").strip() and _normalize_text(str(candidate.get("article") or "")) in query_normalized else 0.0
        hf_priority_boost = 0.05 if str(candidate.get("source_dataset") or "") == "th1nhng0/vietnamese-legal-documents" and int(candidate.get("priority") or 0) == 1 else 0.0
        final_score = (
            float(candidate.get("dense_score") or 0.0) * 0.35
            + float(candidate.get("bm25_score") or 0.0) * 0.25
            + float(candidate.get("exact_score") or 0.0) * 0.20
            + title_overlap * 0.08
            + lexical_overlap * 0.08
            + domain_match * 0.06
            + citation_match * 0.03
            + hf_priority_boost
            - wrong_domain_penalty
        )
        candidate["title_overlap"] = round(title_overlap, 4)
        candidate["lexical_overlap"] = round(lexical_overlap, 4)
        candidate["domain_match"] = round(domain_match, 4)
        candidate["domain_score"] = round(domain_match, 4)
        candidate["citation_match"] = round(citation_match, 4)
        candidate["hf_priority_boost"] = round(hf_priority_boost, 4)
        candidate["wrong_domain_penalty"] = round(wrong_domain_penalty, 4)
        candidate["final_score"] = round(max(0.0, final_score), 6)
        candidate["confidence"] = candidate["final_score"]
        candidate["metadata"] = {
            **dict(candidate.get("metadata") or {}),
            "doc_id": candidate.get("doc_id"),
            "doc_number": candidate.get("doc_number"),
            "doc_title": candidate.get("doc_title"),
            "article": candidate.get("article"),
            "clause": candidate.get("clause"),
            "citation": candidate.get("citation"),
            "domain": candidate.get("domain"),
            "source_url": candidate.get("source_url"),
        }
        reranked.append(candidate)
    reranked.sort(key=lambda item: float(item.get("final_score") or 0.0), reverse=True)

    # Global min-max normalization across fused results
    _min_max_normalize_scores(reranked)

    return reranked[: config.rerank_top_n]


def _min_max_normalize_scores(candidates: list[dict[str, Any]], key: str = "final_score") -> None:
    """In-place min-max normalize a score key to [0, 1] across all candidates."""
    scores = [float(c.get(key, 0)) for c in candidates]
    mn, mx = min(scores), max(scores)
    rng = max(1e-8, mx - mn)
    for c in candidates:
        c[key] = round((float(c.get(key, 0)) - mn) / rng, 6)
        c["confidence"] = c[key]


def _estimate_difficulty(route: str | None, query: str | None) -> str:
    """Estimate question difficulty from route and query heuristics."""
    lowered = _normalize_text(query or "")
    route_str = (route or "SIMPLE_VECTOR").upper()

    # Route-based base difficulty
    route_map = {
        "MULTI_DOMAIN_COMPLEX": "very_hard",
        "LEGAL_GRAPH_CONTEXT": "very_hard",
        "CROSS_DOMAIN_CONTEXT": "hard",
        "PARENT_CONTEXT": "mid",
        "SIMPLE_VECTOR": "easy",
    }
    difficulty = route_map.get(route_str, "easy")

    # Heuristics to bump difficulty
    broad_terms = ["toan bo", "nhung viec gi", "can lam gi", "cac nghia vu", "can nhung van ban nao", "so sanh"]
    if any(t in lowered for t in broad_terms):
        bumps = {"easy": "mid", "mid": "hard", "hard": "very_hard"}
        difficulty = bumps.get(difficulty, difficulty)

    if len(query or "") > 200:
        bumps = {"easy": "mid", "mid": "hard", "hard": "very_hard"}
        difficulty = bumps.get(difficulty, difficulty)

    multi_domain_hints = ["dong thoi", "ket hop", "bao gom ca", "lien quan den", "vua ... vua"]
    if any(t in lowered for t in multi_domain_hints):
        bumps = {"easy": "hard", "mid": "hard", "hard": "very_hard"}
        difficulty = bumps.get(difficulty, difficulty)

    return difficulty


def _difficulty_limits(difficulty: str) -> dict[str, int]:
    limits = {
        "easy": {
            "max_docs": int(os.getenv("R2AI_DIFF_EASY_DOCS", "2")),
            "max_articles": int(os.getenv("R2AI_DIFF_EASY_ARTS", "2")),
            "max_contexts": int(os.getenv("R2AI_DIFF_EASY_CTX", "3")),
            "min_contexts": int(os.getenv("R2AI_DIFF_EASY_MIN", "1")),
        },
        "mid": {
            "max_docs": int(os.getenv("R2AI_DIFF_MID_DOCS", "2")),
            "max_articles": int(os.getenv("R2AI_DIFF_MID_ARTS", "3")),
            "max_contexts": int(os.getenv("R2AI_DIFF_MID_CTX", "5")),
            "min_contexts": int(os.getenv("R2AI_DIFF_MID_MIN", "1")),
        },
        "hard": {
            "max_docs": int(os.getenv("R2AI_DIFF_HARD_DOCS", "4")),
            "max_articles": int(os.getenv("R2AI_DIFF_HARD_ARTS", "10")),
            "max_contexts": int(os.getenv("R2AI_DIFF_HARD_CTX", "12")),
            "min_contexts": int(os.getenv("R2AI_DIFF_HARD_MIN", "2")),
        },
        "very_hard": {
            "max_docs": int(os.getenv("R2AI_DIFF_VERYHARD_DOCS", "5")),
            "max_articles": int(os.getenv("R2AI_DIFF_VERYHARD_ARTS", "15")),
            "max_contexts": int(os.getenv("R2AI_DIFF_VERYHARD_CTX", "15")),
            "min_contexts": int(os.getenv("R2AI_DIFF_VERYHARD_MIN", "3")),
        },
    }
    return limits.get(difficulty, limits["easy"])


def select_dynamic_contexts(
    candidates: Sequence[dict[str, Any]],
    *,
    route: str | None = None,
    query: str | None = None,
    config: RetrievalRuntimeConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or get_retrieval_runtime_config()
    if not candidates:
        return []

    # Difficulty-based limits override config defaults
    difficulty = _estimate_difficulty(route, query)
    limits = _difficulty_limits(difficulty)
    max_docs = limits["max_docs"]
    max_articles = limits["max_articles"]
    max_contexts = limits["max_contexts"]
    min_contexts = limits["min_contexts"]

    selected: list[dict[str, Any]] = []
    doc_count = 0
    article_count = 0
    seen_docs: set[str] = set()
    seen_articles: set[tuple[str, str]] = set()
    # Apply threshold filter (lighter when API-scored)
    has_api_score = any(c.get("api_score") is not None for c in candidates)
    best = float(candidates[0].get("final_score") or 0.0)

    for candidate in candidates:
        score = float(candidate.get("final_score") or 0.0)
        if has_api_score:
            # API already scored: use looser threshold (50% of normal)
            passes = score >= config.absolute_score_threshold * 0.5 or score >= best * config.relative_score_threshold * 0.5
        else:
            passes = score >= config.absolute_score_threshold or score >= best * config.relative_score_threshold
        if not passes:
            continue
        level = str(candidate.get("retrieval_level") or "chunk")
        doc_id = str(candidate.get("doc_id") or candidate.get("doc_number") or "")
        article = str(candidate.get("article") or "")
        if level == "doc":
            if doc_id in seen_docs or doc_count >= max_docs:
                continue
            seen_docs.add(doc_id)
            doc_count += 1
        elif level == "article":
            key = (doc_id, article)
            if key in seen_articles or article_count >= max_articles:
                continue
            seen_articles.add(key)
            article_count += 1
        selected.append(candidate)
        if len(selected) >= max_contexts:
            break
    return selected[: max(min_contexts, min(len(selected), max_contexts))]
