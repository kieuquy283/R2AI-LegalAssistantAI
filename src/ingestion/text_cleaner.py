from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable

from src.ingestion.common import normalize_text, read_jsonl, sha256_text, write_jsonl


DEFAULT_DOCUMENTS_PATH = Path("data/processed/documents.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/cleaned_documents.jsonl")

NOISE_PATTERNS = [
    r"(?im)^ngày cập nhật:.*$",
    r"(?im)^xem thêm.*$",
    r"(?im)^tin liên quan.*$",
    r"(?im)^chia sẻ.*$",
    r"(?im)^mục lục bài viết.*$",
    r"(?im)^tóm tắt luật.*$",
    r"(?im)^văn bản liên quan.*$",
    r"(?im)^đang theo dõi.*$",
]


def _drop_summary_before_legal_body(text: str) -> str:
    markers = [
        r"(?im)^\*{0,2}chương\s+[ivxlcdm0-9]+",
        r"(?im)^\*{0,2}điều\s+\d+[a-z]?\.",
    ]
    starts = []
    for pattern in markers:
        match = re.search(pattern, text)
        if match:
            starts.append(match.start())
    if starts:
        return text[min(starts) :]
    return text


def _strip_markdown_table_lines(text: str) -> str:
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        kept.append(line)
    return "\n".join(kept)


def clean_markdown_text(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = _strip_markdown_table_lines(cleaned)
    for pattern in NOISE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)

    kept_lines = []
    for line in cleaned.splitlines():
        line = re.sub(r"^\s{0,3}[-*]\s+", "- ", line.strip())
        line = re.sub(r"^\*{1,3}(.*?)\*{1,3}$", r"\1", line).strip()
        lowered = line.lower()
        if any(token in lowered for token in ["đăng nhập", "tài khoản", "tiện ích dành cho tài khoản"]):
            continue
        kept_lines.append(line)

    cleaned = normalize_text("\n".join(kept_lines))
    cleaned = _drop_summary_before_legal_body(cleaned)
    cleaned = re.sub(r"(?im)^\#+\s*", "", cleaned)
    cleaned = re.sub(r"(?im)^[ \t]+", "", cleaned)
    return normalize_text(cleaned)


def build_cleaned_record(document: Dict[str, str]) -> Dict[str, str]:
    markdown_path = Path(document["markdown_path"])
    raw_text = markdown_path.read_text(encoding="utf-8")
    cleaned_text = clean_markdown_text(raw_text)
    return {
        "doc_id": document["doc_id"],
        "cleaned_text": cleaned_text,
        "cleaned_text_hash": sha256_text(cleaned_text),
        "source_url": document["source_url"],
        "domain": document["domain"],
        "doc_title": document.get("doc_title"),
    }


def iter_cleaned_documents(documents_path: str | Path) -> Iterable[Dict[str, str]]:
    for row in read_jsonl(documents_path):
        yield build_cleaned_record(row)


def run_text_cleaner(
    *,
    documents_path: str | Path = DEFAULT_DOCUMENTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> int:
    return write_jsonl(output_path, iter_cleaned_documents(documents_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean crawled legal markdown for structural parsing.")
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = run_text_cleaner(documents_path=args.documents, output_path=args.output)
    print(f"Text cleaning: DONE ({count} documents)")


if __name__ == "__main__":
    main()
