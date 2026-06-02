from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from legal_rag.corpus.schema import LegalArticle, normalize_article_number


ARTICLE_LINE_PATTERN = re.compile(r"^(Điều\s+\d+[A-Za-zÀ-ỹăâêôơưĂÂÊÔƠƯđĐ]*)(?:[\.:]\s*(.*))?$", re.MULTILINE)
FULL_NAME_PATTERN = re.compile(r"(Luật|Nghị định|Thông tư|Bộ luật)\s+([^\n]+)")


def extract_article_info(text: str) -> tuple[str, str | None] | None:
    match = ARTICLE_LINE_PATTERN.search(text.strip())
    if not match:
        return None
    article_number = normalize_article_number(match.group(1))
    article_title = (match.group(2) or "").strip() or None
    return article_number, article_title


def extract_doc_identity(doc_id: str, text: str) -> tuple[str, str]:
    full_name_match = FULL_NAME_PATTERN.search(text)
    if full_name_match:
        doc_type = full_name_match.group(1).strip()
        remainder = " ".join(full_name_match.group(2).split())
        short_title = re.sub(r"\s+số\s+[^\s,.;]+.*$", "", remainder, flags=re.IGNORECASE).strip(" ,.;")
        doc_title = f"{doc_type} {short_title}".strip()
        doc_full_name = f"{doc_type} {remainder}".strip()
        return doc_title or doc_id, doc_full_name or doc_title or doc_id
    uppercase_lines = [line.strip() for line in text.splitlines() if line.strip().isupper()]
    if uppercase_lines:
        title = uppercase_lines[-1].title()
        return title, title
    return doc_id, doc_id


def normalize_corpus_items(items: Iterable[dict[str, Any]]) -> list[LegalArticle]:
    items = list(items)
    doc_identity: dict[str, tuple[str, str]] = {}

    for item in items:
        doc_id = str(item.get("doc_id", "")).strip()
        text = str(item.get("content") or item.get("text") or "").strip()
        if not doc_id or not text:
            continue
        if extract_article_info(text) is None and doc_id not in doc_identity:
            doc_identity[doc_id] = extract_doc_identity(doc_id, text)

    articles: list[LegalArticle] = []
    for item in items:
        doc_id = str(item.get("doc_id", "")).strip()
        chunk_id = str(item.get("chunk_id") or item.get("cid") or "").strip()
        chunk_text = str(item.get("content") or item.get("text") or "").strip()
        if not doc_id or not chunk_id or not chunk_text:
            continue

        article_info = extract_article_info(chunk_text)
        if article_info is None:
            continue

        article_number, article_title = article_info
        doc_title, doc_full_name = doc_identity.get(doc_id, (doc_id, doc_id))
        article_id = f"{doc_id}|{doc_title}|{article_number}"
        metadata = item.get("metadata") or {}
        source_path = metadata.get("source_file") or item.get("source_path")

        articles.append(
            LegalArticle(
                doc_id=doc_id,
                doc_title=doc_title,
                doc_full_name=doc_full_name,
                article_id=article_id,
                article_number=article_number,
                article_title=article_title,
                clause_number=None,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                source_path=source_path,
                effective_date=None,
                expiry_date=None,
            )
        )

    return articles


def save_articles_jsonl(path: str | Path, articles: Iterable[LegalArticle]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for article in articles:
            handle.write(json.dumps(article.model_dump(), ensure_ascii=False) + "\n")


def load_articles_jsonl(path: str | Path) -> list[LegalArticle]:
    path = Path(path)
    if not path.exists():
        return []
    articles: list[LegalArticle] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            articles.append(LegalArticle.model_validate(json.loads(stripped)))
    return articles
