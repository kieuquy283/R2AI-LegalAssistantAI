from __future__ import annotations

import re
import json
from typing import Optional

from pydantic import BaseModel, Field, field_validator


ARTICLE_PATTERN = re.compile(r"Điều\s+\d+[A-Za-zÀ-ỹăâêôơưĂÂÊÔƠƯđĐ]*")


def normalize_article_number(value: str) -> str:
    cleaned = " ".join(str(value).strip().split())
    match = ARTICLE_PATTERN.search(cleaned)
    if not match:
        raise ValueError(f"Invalid article number: {value!r}")
    article_number = match.group(0)
    return article_number.replace("đ", "Đ", 1) if article_number.startswith("đ") else article_number


class LegalArticle(BaseModel):
    doc_id: str = Field(..., description="Mã văn bản, ví dụ 04/2017/QH14")
    doc_title: str = Field(..., description="Tên văn bản ngắn")
    doc_full_name: str = Field(..., description="Loại văn bản + mã văn bản + trích yếu")
    article_id: str = Field(..., description="Định danh đầy đủ: doc_id|doc_title|Điều X")
    article_number: str = Field(..., description="Điều X")
    article_title: Optional[str] = None
    clause_number: Optional[str] = None
    chunk_id: str
    chunk_text: str
    source_path: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None

    @field_validator("doc_id", "doc_title", "doc_full_name", "chunk_id", "chunk_text")
    @classmethod
    def ensure_non_empty(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Field must not be empty.")
        return cleaned

    @field_validator("article_number")
    @classmethod
    def validate_article_number(cls, value: str) -> str:
        return normalize_article_number(value)

    @field_validator("article_id")
    @classmethod
    def validate_article_id(cls, value: str) -> str:
        parts = value.split("|")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ValueError("article_id must match <doc_id>|<doc_title>|<Điều X>.")
        normalize_article_number(parts[2])
        return value

    @property
    def doc_ref(self) -> str:
        return f"{self.doc_id}|{self.doc_title}"

    def to_jsonl(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)
