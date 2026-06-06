from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from bs4 import BeautifulSoup

from src.ingestion.common import normalize_text, read_jsonl, write_jsonl


DEFAULT_MANIFEST_PATH = Path("data/raw/documents_manifest.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/documents.jsonl")


def _extract_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", value)
    return match.group(0) if match else None


def _clean_metadata_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = normalize_text(value)
    boilerplate_signals = [
        "ngày ban hành là ngày",
        "cho biết trạng thái hiệu lực",
        "vb liên quan",
        "đang cập nhật",
    ]
    lowered = text.lower()
    if any(signal in lowered for signal in boilerplate_signals):
        extracted_date = _extract_date(text)
        return extracted_date
    return text or None


def _parse_html_fallback(html_path: str | Path) -> Dict[str, Optional[str]]:
    path = Path(html_path)
    if not path.exists():
        return {}

    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    title = soup.find("h1")
    text = normalize_text(soup.get_text("\n", strip=True))
    return {
        "doc_title": normalize_text(title.get_text(" ", strip=True)) if title else None,
        "issue_date": _extract_date(text),
        "effective_date": _extract_date(text[text.find("Hiệu lực") :]) if "Hiệu lực" in text else None,
    }


def build_document_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if not record.get("success"):
        raise ValueError("Only successful manifest records can be parsed")

    html_fallback = _parse_html_fallback(record.get("raw_html_path", ""))
    output = {
        "doc_id": record.get("doc_id"),
        "doc_title": _clean_metadata_value(record.get("doc_title")) or html_fallback.get("doc_title"),
        "doc_number": _clean_metadata_value(record.get("doc_number")),
        "doc_type": _clean_metadata_value(record.get("doc_type")),
        "issuing_body": _clean_metadata_value(record.get("issuing_body")),
        "signer": _clean_metadata_value(record.get("signer")),
        "issue_date": _extract_date(_clean_metadata_value(record.get("issue_date"))) or html_fallback.get("issue_date"),
        "effective_date": _extract_date(_clean_metadata_value(record.get("effective_date"))) or html_fallback.get("effective_date"),
        "status": _clean_metadata_value(record.get("status")),
        "domain": record.get("domain"),
        "source_url": record.get("source_url") or record.get("url"),
        "canonical_url": record.get("canonical_url"),
        "source_id": record.get("source_id"),
        "source_name": record.get("source_name"),
        "provider": record.get("provider"),
        "raw_html_path": record.get("raw_html_path"),
        "markdown_path": record.get("markdown_path"),
        "content_hash": record.get("content_hash"),
    }
    return output


def iter_documents(manifest_rows: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for row in manifest_rows:
        if not row.get("success"):
            continue
        yield build_document_record(row)


def run_document_parser(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> int:
    manifest_rows = read_jsonl(manifest_path)
    return write_jsonl(output_path, iter_documents(manifest_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract document metadata from ingestion manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = run_document_parser(manifest_path=args.manifest, output_path=args.output)
    print(f"Document metadata extraction: DONE ({count} documents)")


if __name__ == "__main__":
    main()
