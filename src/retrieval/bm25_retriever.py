from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.ingestion.common import read_jsonl
from rag.modules.retrieval.utils import tokenize_for_bm25


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d").replace("Đ", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


class _FastBM25:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]]) -> None:
        self.corpus_tokens = [list(tokens) for tokens in corpus_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.k1 = 1.5
        self.b = 0.75
        self.doc_freq: dict[str, int] = {}
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, tokens in enumerate(self.corpus_tokens):
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, freq in counts.items():
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
                self.postings[token].append((doc_idx, freq))

    def get_scores(self, query_tokens: Sequence[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        total_docs = max(len(self.corpus_tokens), 1)
        for token in query_tokens:
            postings = self.postings.get(token)
            if not postings:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
            for doc_idx, freq in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avgdl, 1e-8)))
                scores[doc_idx] += idf * ((freq * (self.k1 + 1)) / max(denom, 1e-8))
        return scores


class BM25Retriever:
    def __init__(self, *, chunks_path: str | Path | None = None) -> None:
        self.chunks_path = Path(chunks_path or self._default_chunks_path())
        self.rows = read_jsonl(self.chunks_path)
        self.chunk_ids = [str(row.get("chunk_id") or "") for row in self.rows]
        corpus_tokens = [tokenize_for_bm25(_normalize_text(self._combined_text(row))) for row in self.rows]
        self.bm25 = _FastBM25(corpus_tokens)

    @staticmethod
    def _default_chunks_path() -> Path:
        overridden = os.getenv("R2AI_CHUNKS_PATH", "").strip()
        if overridden:
            return Path(overridden)
        merged = Path("data/processed/merged_chunks.jsonl")
        if merged.exists():
            return merged
        return Path("data/processed/chunks.jsonl")

    @staticmethod
    def _combined_text(row: dict[str, Any]) -> str:
        parts = [
            row.get("doc_title"),
            row.get("doc_number"),
            row.get("citation"),
            row.get("article"),
            row.get("clause"),
            row.get("content"),
            row.get("domain"),
        ]
        return "\n".join(str(part or "").strip() for part in parts if str(part or "").strip())

    @staticmethod
    def _allowed_domain(row: dict[str, Any], preferred_domains: Sequence[str] | None) -> bool:
        return True

    @staticmethod
    def _normalize_scores(values: list[float]) -> list[float]:
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if maximum == minimum:
            return [1.0 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def search(self, query: str, *, top_k: int, preferred_domains: Sequence[str] | None = None) -> list[dict[str, Any]]:
        query_tokens = tokenize_for_bm25(_normalize_text(query))
        if not query_tokens:
            return []
        raw_scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)[: max(top_k * 3, top_k)]
        normalized = self._normalize_scores([float(score) for _idx, score in ranked])
        candidates: list[dict[str, Any]] = []
        for (idx, _score), normalized_score in zip(ranked, normalized, strict=True):
            if idx >= len(self.rows):
                continue
            row = dict(self.rows[idx])
            if not self._allowed_domain(row, preferred_domains):
                continue
            candidates.append(
                {
                    "candidate_id": f"chunk:{row.get('chunk_id')}",
                    "retrieval_level": "chunk",
                    "retrieval_method": "bm25",
                    "dense_score": 0.0,
                    "bm25_score": float(normalized_score),
                    "exact_score": 0.0,
                    "title_overlap": 0.0,
                    "lexical_overlap": 0.0,
                    "domain_match": 0.0,
                    "domain_score": 0.0,
                    "citation_match": 0.0,
                    "final_score": float(normalized_score),
                    "confidence": float(normalized_score),
                    "chunk_id": str(row.get("chunk_id") or ""),
                    "doc_id": str(row.get("doc_id") or ""),
                    "article_id": str(row.get("node_id") or ""),
                    "chunk_ref_id": str(row.get("chunk_id") or ""),
                    "doc_number": str(row.get("doc_number") or ""),
                    "doc_title": str(row.get("doc_title") or ""),
                    "article": str(row.get("article") or ""),
                    "clause": str(row.get("clause") or ""),
                    "citation": str(row.get("citation") or row.get("doc_title") or ""),
                    "domain": str(row.get("domain") or ""),
                    "source_url": str(row.get("source_url") or ""),
                    "content": str(row.get("content") or ""),
                    "source_dataset": str(row.get("source_dataset") or "local_corpus"),
                    "priority": int(row.get("priority") or 0),
                    "metadata": row,
                }
            )
            if len(candidates) >= top_k:
                break
        return candidates
