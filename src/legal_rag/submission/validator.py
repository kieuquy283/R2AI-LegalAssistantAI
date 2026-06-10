from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from legal_rag.submission.schema import SubmissionItem
from legal_rag.utils import load_json


DOC_PATTERN = re.compile(r"^[^|]+\|[^|]+$")
ARTICLE_REF_PATTERN = re.compile(r"^[^|]+\|[^|]+\|Điều\s+\d+[A-Za-zÀ-ỹăâêôơưĂÂÊÔƠƯđĐ]*$")


class SubmissionValidationReport(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    item_count: int


def validate_submission_payload(payload: Any) -> SubmissionValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(payload, list):
        return SubmissionValidationReport(ok=False, errors=["Submission JSON must be a list."], warnings=[], item_count=0)

    seen_ids: set[int] = set()
    for index, raw_item in enumerate(payload):
        try:
            item = SubmissionItem.model_validate(raw_item)
        except Exception as exc:  # pragma: no cover
            errors.append(f"Item {index} is invalid: {exc}")
            continue

        if item.id in seen_ids:
            errors.append(f"Duplicate id detected: {item.id}")
        seen_ids.add(item.id)

        if any(not DOC_PATTERN.match(doc_ref) for doc_ref in item.relevant_docs):
            errors.append(f"Item {item.id} has invalid relevant_docs format.")

        if any(not ARTICLE_REF_PATTERN.match(article_ref) for article_ref in item.relevant_articles):
            errors.append(f"Item {item.id} has invalid relevant_articles format.")

        if len(set(item.relevant_articles)) != len(item.relevant_articles):
            errors.append(f"Item {item.id} has duplicate relevant_articles entries.")

    return SubmissionValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        item_count=len(payload),
    )


def validate_submission_file(path: str | Path) -> SubmissionValidationReport:
    payload = load_json(path, default=None)
    return validate_submission_payload(payload)
