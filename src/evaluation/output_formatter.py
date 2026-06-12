from __future__ import annotations

import re
from typing import Any


LEGAL_CODE_PATTERN = re.compile(r"\b\d+(?:/\d+)+/[A-Z0-9À-ỴĂÂĐÊÔƠƯ\-]+\b", re.IGNORECASE)


def extract_legal_doc_code(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        match = LEGAL_CODE_PATTERN.search(text)
        if match:
            return match.group(0)
    return ""


def format_doc_ref(candidate: dict[str, Any]) -> str:
    doc_id = str(candidate.get("doc_id") or "").strip()
    doc_title = str(candidate.get("doc_title") or "").strip()
    citation = str(candidate.get("citation") or "").strip()
    left = extract_legal_doc_code(doc_title, citation, doc_id) or doc_id or doc_title
    right = doc_title or doc_id
    if not left or not right:
        return ""
    return f"{left}|{right}"


def format_article_ref(candidate: dict[str, Any]) -> str:
    base = format_doc_ref(candidate)
    article = str(candidate.get("article") or "").strip()
    if not base or not article:
        return ""
    return f"{base}|{article}"


def format_submission_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(result["id"]),
        "question": str(result.get("question") or ""),
        "answer": str(result.get("answer") or ""),
        "relevant_docs": [str(item) for item in list(result.get("relevant_docs") or []) if str(item).count("|") == 1],
        "relevant_articles": [str(item) for item in list(result.get("relevant_articles") or []) if str(item).count("|") >= 2],
    }
