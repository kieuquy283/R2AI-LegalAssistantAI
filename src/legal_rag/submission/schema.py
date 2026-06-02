from __future__ import annotations

from pydantic import BaseModel, field_validator


class SubmissionItem(BaseModel):
    id: int
    question: str
    answer: str
    relevant_docs: list[str]
    relevant_articles: list[str]

    @field_validator("question", "answer")
    @classmethod
    def ensure_non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field must not be empty.")
        return cleaned
