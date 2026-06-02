from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel

from legal_rag.aggregation.article import SelectedArticle
from legal_rag.generation.citation_validator import validate_citations


class AnswerQualityReport(BaseModel):
    ok: bool
    legal_basis_ok: bool
    content_warnings: list[str]
    completeness_warnings: list[str]
    practicality_ok: bool
    clarity_ok: bool


def check_answer_quality(question: str, answer: str, selected_articles: Iterable[SelectedArticle]) -> AnswerQualityReport:
    articles = list(selected_articles)
    citation_report = validate_citations(answer, articles)
    content_warnings: list[str] = []
    completeness_warnings: list[str] = []

    if "theo luật hiện hành" in answer.lower() and "Điều " not in answer:
        content_warnings.append("Generic legal claim without explicit article citation.")

    query_lower = question.lower()
    if any(keyword in query_lower for keyword in ("điều kiện", "thủ tục", "nghĩa vụ", "quyền", "xử phạt")) and len(articles) < 2:
        completeness_warnings.append("Broad legal question may require more than one article.")

    practicality_ok = "Gợi ý áp dụng thực tế" in answer
    clarity_ok = all(section in answer for section in ("Kết luận ngắn", "Căn cứ pháp lý", "Giải thích dễ hiểu", "Lưu ý"))
    ok = citation_report.ok and not content_warnings and practicality_ok and clarity_ok

    return AnswerQualityReport(
        ok=ok,
        legal_basis_ok=citation_report.ok,
        content_warnings=content_warnings,
        completeness_warnings=completeness_warnings,
        practicality_ok=practicality_ok,
        clarity_ok=clarity_ok,
    )
