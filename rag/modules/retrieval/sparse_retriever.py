from __future__ import annotations

import math
from typing import List, Optional

from .base import BaseRetriever
from .schemas import RetrievalResult
from .utils import (
    deduplicate_results,
    filter_active_results,
    normalize_sparse_scores,
    tokenize_for_bm25,
)


class SparseRetriever(BaseRetriever):
    """
    Sparse retriever using BM25.
    """

    def __init__(
        self,
        documents,
        top_k: int = 5,
        candidate_k: Optional[int] = None,
        filter_active: bool = True,
    ) -> None:
        self.documents = list(documents or [])
        self.top_k = int(top_k)
        self.candidate_k = int(candidate_k or max(self.top_k * 2, self.top_k + 5))
        self.filter_active = filter_active

        self.tokenized_corpus = [
            tokenize_for_bm25(getattr(doc, "page_content", ""))
            for doc in self.documents
        ]

        if self.documents and any(self.tokenized_corpus):
            try:
                from rank_bm25 import BM25Okapi

                self.bm25 = BM25Okapi(self.tokenized_corpus)
            except ModuleNotFoundError:
                self.bm25 = _SimpleBM25(self.tokenized_corpus)
        else:
            self.bm25 = None

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        if not query.strip() or not self.documents or self.bm25 is None:
            return []

        tokenized_query = tokenize_for_bm25(query)
        if not tokenized_query:
            return []

        result_top_k = int(top_k or self.top_k)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[: self.candidate_k]

        results: List[RetrievalResult] = []
        for rank, index in enumerate(ranked_indices, start=1):
            doc = self.documents[index]
            metadata = dict(getattr(doc, "metadata", {}) or {})
            text = str(getattr(doc, "page_content", "") or "").strip()
            if not text:
                continue

            chunk_id = str(metadata.get("chunk_id", index))
            result = RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                score=float(scores[index]),
                source="sparse",
                metadata=metadata,
                retrieval_rank=rank,
                raw_score=float(scores[index]),
                retriever_name="bm25",
                sources=["sparse"],
            )
            result.metadata["retrieval_sources"] = ["sparse"]
            results.append(result)

        results = normalize_sparse_scores(results)
        if self.filter_active:
            results, _ = filter_active_results(results)
        results = deduplicate_results(results)
        return results[:result_top_k]


class _SimpleBM25:
    def __init__(self, corpus_tokens: List[List[str]]) -> None:
        self.corpus_tokens = corpus_tokens
        self.doc_freq = {}
        self.doc_lengths = [len(tokens) for tokens in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.k1 = 1.5
        self.b = 0.75

        for tokens in corpus_tokens:
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        num_docs = max(len(self.corpus_tokens), 1)
        scores: List[float] = []

        for tokens in self.corpus_tokens:
            term_counts = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            doc_length = len(tokens)
            score = 0.0

            for token in query_tokens:
                freq = term_counts.get(token, 0)
                if freq == 0:
                    continue

                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + ((num_docs - df + 0.5) / (df + 0.5)))
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * (doc_length / max(self.avgdl, 1e-8))
                )
                score += idf * ((freq * (self.k1 + 1)) / max(denominator, 1e-8))

            scores.append(score)

        return scores
