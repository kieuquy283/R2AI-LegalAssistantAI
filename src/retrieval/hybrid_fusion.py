from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

from rag.config.runtime import RetrievalRuntimeConfig, get_retrieval_runtime_config
from rag.modules.retrieval.utils import tokenize_for_bm25


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


def _merge_candidates(candidates: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = str(candidate.get("candidate_id") or "")
        if not key:
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(candidate)
            continue
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


def fuse_candidates(
    query: str,
    *,
    dense_candidates: Sequence[dict[str, Any]],
    bm25_candidates: Sequence[dict[str, Any]],
    exact_candidates: Sequence[dict[str, Any]],
    preferred_domains: Sequence[str] | None = None,
    config: RetrievalRuntimeConfig | None = None,
) -> list[dict[str, Any]]:
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
    return reranked[: config.rerank_top_n]


def select_dynamic_contexts(
    candidates: Sequence[dict[str, Any]],
    *,
    config: RetrievalRuntimeConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or get_retrieval_runtime_config()
    if not candidates:
        return []
    best = float(candidates[0].get("final_score") or 0.0)
    selected: list[dict[str, Any]] = []
    doc_count = 0
    article_count = 0
    seen_docs: set[str] = set()
    seen_articles: set[tuple[str, str]] = set()
    for candidate in candidates:
        score = float(candidate.get("final_score") or 0.0)
        passes = score >= config.absolute_score_threshold or score >= best * config.relative_score_threshold
        if not passes:
            continue
        level = str(candidate.get("retrieval_level") or "chunk")
        doc_id = str(candidate.get("doc_id") or candidate.get("doc_number") or "")
        article = str(candidate.get("article") or "")
        if level == "doc":
            if doc_id in seen_docs or doc_count >= config.max_docs:
                continue
            seen_docs.add(doc_id)
            doc_count += 1
        elif level == "article":
            key = (doc_id, article)
            if key in seen_articles or article_count >= config.max_articles:
                continue
            seen_articles.add(key)
            article_count += 1
        selected.append(candidate)
        if len(selected) >= config.max_contexts:
            break
    return selected[: max(config.min_contexts, min(len(selected), config.max_contexts))]
