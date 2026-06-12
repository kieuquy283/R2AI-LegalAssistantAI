from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from rag.modules.retrieval.utils import tokenize_for_bm25
from src.ingestion.common import read_jsonl


LEGAL_REF_PATTERN = re.compile(r"\b\d+(?:/\d+)+/[A-Z0-9À-ỴĂÂĐÊÔƠƯ\-]+\b", re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"(điều|dieu)\s+\d+[A-Za-zÀ-ỴăâêôơưĐđ]*", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d").replace("Đ", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


class LegalExactSearch:
    def __init__(
        self,
        *,
        documents_path: str | Path | None = None,
        articles_path: str | Path | None = None,
        chunks_path: str | Path | None = None,
    ) -> None:
        self.documents = read_jsonl(documents_path or self._default_path("merged_documents.jsonl", "documents.jsonl"))
        self.articles = read_jsonl(articles_path or self._default_path("merged_legal_nodes.jsonl", "legal_nodes.jsonl"))
        self.chunks = read_jsonl(chunks_path or self._default_path("merged_chunks.jsonl", "chunks.jsonl"))

    @staticmethod
    def _default_path(merged_name: str, base_name: str) -> Path:
        override_map = {
            "merged_documents.jsonl": os.getenv("R2AI_DOCUMENTS_PATH", "").strip(),
            "merged_legal_nodes.jsonl": os.getenv("R2AI_ARTICLES_PATH", "").strip(),
            "merged_chunks.jsonl": os.getenv("R2AI_CHUNKS_PATH", "").strip(),
        }
        overridden = override_map.get(merged_name, "")
        if overridden:
            return Path(overridden)
        merged = Path("data/processed") / merged_name
        if merged.exists():
            return merged
        return Path("data/processed") / base_name

    @staticmethod
    def _allowed_domain(row: dict[str, Any], preferred_domains: Sequence[str] | None) -> bool:
        return True

    @staticmethod
    def _score(query: str, row: dict[str, Any], level: str) -> float:
        normalized_query = _normalize_text(query)
        query_tokens = set(tokenize_for_bm25(normalized_query))
        title = str(row.get("doc_title") or row.get("title") or "").strip()
        doc_number = str(row.get("doc_number") or "").strip()
        article = str(row.get("article") or "").strip()
        citation = str(row.get("citation") or title).strip()
        content = str(row.get("content") or "").strip()
        haystack = _normalize_text("\n".join([doc_number, title, article, citation, content]))
        title_tokens = set(tokenize_for_bm25(_normalize_text(" ".join([title, citation, article]))))
        lexical_overlap = len(query_tokens & set(tokenize_for_bm25(haystack))) / max(len(query_tokens), 1) if query_tokens else 0.0
        title_overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
        legal_ref_score = 1.0 if doc_number and LEGAL_REF_PATTERN.search(query) and doc_number in query else 0.0
        article_query = ARTICLE_PATTERN.search(query)
        article_score = 1.0 if article and article_query and _normalize_text(article_query.group(0)) in _normalize_text(article) else 0.0
        if not any([legal_ref_score, article_score, title_overlap, lexical_overlap]):
            return 0.0
        level_boost = 0.05 if level == "article" else 0.0
        return min(1.0, legal_ref_score * 0.6 + article_score * 0.2 + title_overlap * 0.15 + lexical_overlap * 0.1 + level_boost)

    @staticmethod
    def _make_candidate(level: str, row: dict[str, Any], score: float) -> dict[str, Any]:
        candidate_id = str(row.get("chunk_id") or row.get("node_id") or row.get("doc_id") or "")
        return {
            "candidate_id": f"{level}:{candidate_id}",
            "retrieval_level": level,
            "retrieval_method": "exact",
            "dense_score": 0.0,
            "bm25_score": 0.0,
            "exact_score": float(score),
            "title_overlap": float(score),
            "lexical_overlap": float(score),
            "domain_match": 0.0,
            "domain_score": 0.0,
            "citation_match": 1.0 if str(row.get("article") or "").strip() else 0.0,
            "final_score": float(score),
            "confidence": float(score),
            "chunk_id": str(row.get("chunk_id") or candidate_id),
            "doc_id": str(row.get("doc_id") or ""),
            "article_id": str(row.get("node_id") or ""),
            "chunk_ref_id": str(row.get("chunk_id") or ""),
            "doc_number": str(row.get("doc_number") or ""),
            "doc_title": str(row.get("doc_title") or row.get("title") or ""),
            "article": str(row.get("article") or ""),
            "clause": str(row.get("clause") or ""),
            "citation": str(row.get("citation") or row.get("title") or row.get("doc_title") or ""),
            "domain": str(row.get("domain") or ""),
            "source_url": str(row.get("source_url") or ""),
            "content": str(row.get("content") or row.get("cleaned_text") or ""),
            "source_dataset": str(row.get("source_dataset") or "local_corpus"),
            "priority": int(row.get("priority") or 0),
            "metadata": dict(row),
        }

    def search(self, query: str, *, top_k: int, preferred_domains: Sequence[str] | None = None) -> list[dict[str, Any]]:
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for level, rows in (("doc", self.documents), ("article", self.articles), ("chunk", self.chunks)):
            for row in rows:
                if not self._allowed_domain(row, preferred_domains):
                    continue
                score = self._score(query, row, level)
                if score > 0.0:
                    scored.append((score, level, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._make_candidate(level, row, score) for score, level, row in scored[:top_k]]
