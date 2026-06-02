from __future__ import annotations

import re
from typing import Any, Dict, List


FOLLOW_UP_MARKERS = {
    "vậy",
    "còn",
    "trường hợp đó",
    "quy định này",
    "luật này",
    "điều đó",
    "nó",
    "như vậy",
    "ở đây",
    "trong trường hợp này",
}

LEGAL_KEYWORDS = {
    "điều",
    "khoản",
    "luật",
    "nghị quyết",
    "nghị định",
    "thông tư",
    "thẩm quyền",
    "hiệu lực",
    "ban hành",
    "quy phạm pháp luật",
}

ANAPHORA_MARKERS = {
    "đó",
    "này",
    "nó",
    "cái đó",
    "trường hợp đó",
    "quy định này",
    "luật này",
    "điều đó",
    "như vậy",
}

ARTICLE_REFERENCE_RE = re.compile(r"\b(điều|khoản|điểm)\s+\d+[a-zA-Z0-9/-]*", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _contains_any(text: str, markers: set[str]) -> bool:
    return any(marker in text for marker in markers)


def analyze_query(question: str, history: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    normalized_question = _normalize_text(question)
    history = list(history or [])
    question_tokens = normalized_question.split()
    question_length = len(question_tokens)

    has_followup_markers = _contains_any(normalized_question, FOLLOW_UP_MARKERS)
    has_anaphora = _contains_any(normalized_question, ANAPHORA_MARKERS)
    has_legal_keywords = _contains_any(normalized_question, LEGAL_KEYWORDS)
    has_article_reference = bool(ARTICLE_REFERENCE_RE.search(normalized_question))

    history_available = bool(history)
    needs_history = history_available and (has_followup_markers or has_anaphora)
    is_standalone = not needs_history

    complexity_markers = sum(
        [
            1 if question_length >= 12 else 0,
            1 if normalized_question.count("?") >= 2 else 0,
            1 if "và" in question_tokens or "hoặc" in question_tokens else 0,
            1 if has_article_reference else 0,
            1 if normalized_question.count(",") >= 2 else 0,
        ]
    )
    is_complex = complexity_markers >= 2 or question_length >= 18
    is_very_complex = complexity_markers >= 4 or question_length >= 28

    abstract_markers = {"thế nào", "là gì", "giải thích", "ý nghĩa", "nguyên tắc"}
    is_abstract = _contains_any(normalized_question, abstract_markers) and not has_article_reference

    return {
        "is_standalone": is_standalone,
        "needs_history": needs_history,
        "has_anaphora": has_anaphora,
        "is_complex": is_complex,
        "is_very_complex": is_very_complex,
        "is_abstract": is_abstract,
        "question_length": question_length,
        "has_legal_keywords": has_legal_keywords,
        "has_article_reference": has_article_reference,
        "has_followup_markers": has_followup_markers,
        "history_turn_count": len(history),
    }

