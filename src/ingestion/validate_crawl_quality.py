from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .common import read_jsonl, write_json


DEFAULT_MANIFEST_PATH = Path("data/raw/documents_manifest.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/raw/crawl_quality_report.json")


LEGAL_SIGNAL_TERMS = [
    "Điều ",
    "Khoản ",
    "Căn cứ",
    "Luật ",
    "Nghị định",
    "Thông tư",
]


def _is_nonempty_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _pick_success_rows(rows: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    success_rows = [row for row in rows if row.get("success")]
    return success_rows[:limit]


def validate_crawl_quality(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    limit: int = 500,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)

    rows = read_jsonl(manifest_path)
    success_rows = _pick_success_rows(rows, limit=limit)

    empty_html: List[str] = []
    empty_markdown: List[str] = []
    short_markdown: List[str] = []
    missing_paths: List[str] = []
    legal_signal_count = 0
    restricted_signal_count = 0
    domains = Counter()

    for row in success_rows:
        domains[row.get("domain", "unknown")] += 1

        html_path = Path(row.get("raw_html_path", ""))
        markdown_path = Path(row.get("markdown_path", ""))
        source_url = row.get("source_url") or row.get("canonical_url") or row.get("url")

        if not html_path.exists() or not markdown_path.exists():
            missing_paths.append(source_url)
            continue

        html = _read_text_file(html_path)
        markdown = _read_text_file(markdown_path)

        if not html:
            empty_html.append(source_url)
        if not markdown:
            empty_markdown.append(source_url)
        elif len(markdown) < 500:
            short_markdown.append(source_url)

        if any(term in markdown for term in LEGAL_SIGNAL_TERMS):
            legal_signal_count += 1

        access = row.get("access_restriction", {}) or {}
        if access.get("has_restriction_signal"):
            restricted_signal_count += 1

    report = {
        "manifest_path": str(manifest_path),
        "output_path": str(output_path),
        "manifest_rows": len(rows),
        "manifest_success_rows": len([row for row in rows if row.get("success")]),
        "sampled_success_rows": len(success_rows),
        "missing_paths": len(missing_paths),
        "empty_html": len(empty_html),
        "empty_markdown": len(empty_markdown),
        "short_markdown_under_500": len(short_markdown),
        "legal_signal_count": legal_signal_count,
        "restricted_signal_count": restricted_signal_count,
        "domains": dict(domains),
        "sample_empty_markdown": empty_markdown[:5],
        "sample_short_markdown": short_markdown[:5],
    }

    write_json(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate crawl quality for sampled legal documents.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--limit", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_crawl_quality(
        manifest_path=args.manifest,
        output_path=args.output,
        limit=args.limit,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    assert report["sampled_success_rows"] >= 500, (
        f"Expected at least 500 sampled success rows, got {report['sampled_success_rows']}"
    )
    assert report["missing_paths"] == 0, f"Missing file paths: {report['missing_paths']}"
    assert report["empty_html"] < max(report["sampled_success_rows"] * 0.1, 1), "Too many empty HTML files"
    assert report["empty_markdown"] < max(report["sampled_success_rows"] * 0.2, 1), "Too many empty Markdown files"
    assert report["legal_signal_count"] > 0, "No legal content signal found"

    print("CRAWL QUALITY VALIDATION PASSED")


if __name__ == "__main__":
    main()
