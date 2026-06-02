from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from pydantic import BaseModel

from legal_rag.corpus.schema import LegalArticle
from legal_rag.retrieval.keyword import RetrievedChunk


class SelectedArticle(BaseModel):
    article_id: str
    article_number: str
    doc_id: str
    doc_title: str
    doc_full_name: str
    article_title: str | None = None
    score: float
    chunk_ids: list[str]
    evidence: list[str]

    @property
    def doc_ref(self) -> str:
        return f"{self.doc_id}|{self.doc_title}"


class ArticleAggregator:
    def __init__(
        self,
        *,
        strategy: str = "dynamic_threshold",
        default_top_k: int = 5,
        min_articles: int = 1,
        max_articles: int = 7,
        score_threshold: float = 0.35,
        relative_threshold: float = 0.75,
    ) -> None:
        self.strategy = strategy
        self.default_top_k = default_top_k
        self.min_articles = min_articles
        self.max_articles = max_articles
        self.score_threshold = score_threshold
        self.relative_threshold = relative_threshold

    def select(self, retrieved_chunks: Iterable[RetrievedChunk], *, query: str = "") -> list[SelectedArticle]:
        grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in retrieved_chunks:
            grouped[chunk.article_id].append(chunk)

        selected: list[SelectedArticle] = []
        for article_id, chunks in grouped.items():
            chunks.sort(key=self._score_chunk, reverse=True)
            article: LegalArticle = chunks[0].article
            article_score = max(self._score_chunk(chunk) for chunk in chunks)
            selected.append(
                SelectedArticle(
                    article_id=article_id,
                    article_number=article.article_number,
                    doc_id=article.doc_id,
                    doc_title=article.doc_title,
                    doc_full_name=article.doc_full_name,
                    article_title=article.article_title,
                    score=article_score,
                    chunk_ids=[chunk.chunk_id for chunk in chunks],
                    evidence=[chunk.text for chunk in chunks[:2]],
                )
            )

        selected.sort(key=lambda item: item.score, reverse=True)
        return self._trim(selected, query=query)

    def _score_chunk(self, chunk: RetrievedChunk) -> float:
        components = [
            chunk.rerank_score,
            chunk.dense_score,
            chunk.sparse_score,
            chunk.score,
        ]
        usable = [value for value in components if value is not None]
        return max(usable) if usable else 0.0

    def _trim(self, articles: list[SelectedArticle], *, query: str) -> list[SelectedArticle]:
        if not articles:
            return []
        if self.strategy != "dynamic_threshold":
            return articles[: self.default_top_k]

        top_score = articles[0].score
        query_lower = query.lower()
        broad_query = any(keyword in query_lower for keyword in ("điều kiện", "thủ tục", "nghĩa vụ", "quyền", "xử phạt"))
        target_max = min(self.max_articles, self.default_top_k + (1 if broad_query else 0))

        selected = [
            article
            for article in articles
            if article.score >= self.score_threshold and article.score >= top_score * self.relative_threshold
        ]
        if len(selected) < self.min_articles:
            selected = articles[: self.min_articles]
        return selected[:target_max]
