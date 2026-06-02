from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel

from legal_rag.aggregation.article import SelectedArticle


ARTICLE_PATTERN = r"Điều\s+\d+[a-zA-ZăâêôơưĂÂÊÔƠƯĐđ]*"


class CitationValidationResult(BaseModel):
    ok: bool
    cited_articles: list[str]
    missing_articles: list[str]
    unsupported_citations: list[str]
    warnings: list[str]


def extract_cited_articles(answer: str) -> list[str]:
    return list(dict.fromkeys(re.findall(ARTICLE_PATTERN, answer)))


def validate_citations(answer: str, selected_articles: Iterable[SelectedArticle]) -> CitationValidationResult:
    selected_numbers = [article.article_number for article in selected_articles]
    cited_articles = extract_cited_articles(answer)
    missing = [article for article in selected_numbers if article not in cited_articles]
    unsupported = [article for article in cited_articles if article not in selected_numbers]
    warnings: list[str] = []
    if not cited_articles:
        warnings.append("Answer does not cite any legal article.")
    if unsupported:
        warnings.append("Answer cites articles outside selected evidence.")
    return CitationValidationResult(
        ok=not missing and not unsupported and bool(cited_articles),
        cited_articles=cited_articles,
        missing_articles=missing,
        unsupported_citations=unsupported,
        warnings=warnings,
    )


def ensure_citations(answer: str, selected_articles: Iterable[SelectedArticle]) -> str:
    articles = list(selected_articles)
    result = validate_citations(answer, articles)
    if result.ok or not articles:
        return answer
    legal_basis = "; ".join(f"{article.article_number} {article.doc_title}" for article in articles[:3])
    prefix = f"Căn cứ pháp lý: Căn cứ {legal_basis}.\n\n"
    return prefix + answer
