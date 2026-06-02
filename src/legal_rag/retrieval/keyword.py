from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from legal_rag.corpus.schema import LegalArticle


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class RetrievedChunk:
    article: LegalArticle
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.article.chunk_id

    @property
    def article_id(self) -> str:
        return self.article.article_id

    @property
    def text(self) -> str:
        return self.article.chunk_text


class KeywordArticleRetriever:
    def __init__(self, articles: Iterable[LegalArticle]) -> None:
        self.articles = list(articles)

    def retrieve(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_token_set = set(query_tokens)
        scored: list[RetrievedChunk] = []
        for article in self.articles:
            title_text = " ".join(filter(None, [article.article_title or "", article.article_number]))
            title_tokens = tokenize(title_text)
            body_tokens = tokenize(article.chunk_text)
            article_text = " ".join(
                filter(
                    None,
                    [
                        article.doc_title,
                        article.article_number,
                        article.article_title or "",
                        article.chunk_text,
                    ],
                )
            )
            article_tokens = tokenize(article_text)
            if not article_tokens:
                continue
            title_overlap = len(query_token_set & set(title_tokens))
            body_overlap = len(query_token_set & set(body_tokens))
            if title_overlap == 0 and body_overlap == 0:
                continue
            exact_title_bonus = 2.0 if article.article_title and article.article_title.lower() in query.lower() else 0.0
            positional_bonus = 0.5 if article.article_number in article_text else 0.0
            score = (
                (2.5 * title_overlap) / max(len(query_token_set), 1)
                + (0.75 * body_overlap) / max(len(query_token_set), 1)
                + exact_title_bonus
                + positional_bonus
            )
            scored.append(
                RetrievedChunk(
                    article=article,
                    score=score,
                    dense_score=score,
                    sparse_score=(title_overlap + body_overlap) / max(len(article_tokens), 1),
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
