from __future__ import annotations

import re
from pathlib import Path
from typing import List

from rag.config.retrieval import CHUNK_OVERLAP, CHUNK_SIZE


class SimpleRecursiveTextSplitter:
    """Small local fallback to avoid hard import-time dependency on LangChain splitters."""

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str],
    ) -> None:
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        self.separators = separators

    def split_text(self, text: str) -> List[str]:
        normalized = text.strip()
        if not normalized:
            return []

        pieces = self._split_recursive(normalized, self.separators)
        chunks: List[str] = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            candidate = piece if not current else f"{current} {piece}".strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
            current = piece

        if current:
            chunks.append(current)

        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        overlapped: List[str] = [chunks[0]]
        for chunk in chunks[1:]:
            prefix = overlapped[-1][-self.chunk_overlap :].strip()
            overlapped.append(f"{prefix} {chunk}".strip() if prefix else chunk)
        return overlapped

    def _split_recursive(self, text: str, separators: list[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = separators[0]
        if separator:
            parts = [part.strip() for part in text.split(separator) if part and part.strip()]
        else:
            parts = [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        if len(parts) <= 1:
            return self._split_recursive(text, separators[1:])

        flattened: List[str] = []
        for part in parts:
            flattened.extend(self._split_recursive(part, separators[1:]))
        return flattened


def split_legal_articles(text: str) -> List[str]:
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []

    pattern = re.compile(
        r"(?im)^\s*(Điều\s+\d+\s*[\.:]?\s*.*?)(?=^\s*Điều\s+\d+\s*[\.:]?\s*|\Z)",
        re.DOTALL,
    )
    matches = pattern.findall(normalized)
    return [m.strip() for m in matches if m and m.strip()]


def build_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> SimpleRecursiveTextSplitter:
    return SimpleRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def fallback_text_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    splitter = build_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c and c.strip()]


def chunk_legal_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    article_chunks = split_legal_articles(text)

    if len(article_chunks) >= 2:
        final_chunks: List[str] = []
        for chunk in article_chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(
                    fallback_text_chunks(
                        chunk,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
        return [c for c in final_chunks if c.strip()]

    return fallback_text_chunks(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_document(
    file_path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    from rag.ingestion.readers import read_document

    text = read_document(file_path)
    return chunk_legal_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
