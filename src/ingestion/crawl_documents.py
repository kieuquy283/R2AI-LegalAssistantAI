"""
Crawl legal document detail pages and save raw HTML, Markdown, metadata manifest.

This module is Task 3 in the ingestion pipeline.

Input:
    data/raw/document_urls.jsonl

Output:
    data/raw/html/*.html
    data/raw/markdown/*.md
    data/raw/documents_manifest.jsonl
    data/raw/crawl_documents_report.json
    data/logs/crawl_documents_errors.jsonl

Usage:
    python -m src.ingestion.crawl_documents \
        --input data/raw/document_urls.jsonl \
        --limit 10

Notes:
    - This module crawls public detail pages only.
    - It does not bypass login, paywall, CAPTCHA, or access restrictions.
    - It prefers public HTML/Markdown content over downloading files.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", str(Path.cwd() / "data"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path.cwd() / "data" / "playwright-browsers"))

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter
except Exception:  # pragma: no cover - handled at runtime
    AsyncWebCrawler = None
    BrowserConfig = None
    CrawlerRunConfig = None
    CacheMode = None
    DefaultMarkdownGenerator = None
    PruningContentFilter = None


DEFAULT_INPUT_PATH = Path("data/raw/document_urls.jsonl")
DEFAULT_HTML_DIR = Path("data/raw/html")
DEFAULT_MARKDOWN_DIR = Path("data/raw/markdown")
DEFAULT_MANIFEST_PATH = Path("data/raw/documents_manifest.jsonl")
DEFAULT_REPORT_PATH = Path("data/raw/crawl_documents_report.json")
DEFAULT_ERROR_LOG_PATH = Path("data/logs/crawl_documents_errors.jsonl")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def remove_vietnamese_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    no_accents = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )
    return no_accents.replace("đ", "d").replace("Đ", "D")


def slugify(text: str, max_length: int = 160) -> str:
    text = remove_vietnamese_accents(text or "")
    text = text.lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "document")[:max_length]


def stable_doc_id(record: Dict[str, Any], html: str = "") -> str:
    """
    Build a stable doc_id from doc_number/title if possible, otherwise URL hash.
    """
    metadata = extract_document_metadata(html, record.get("url", "")) if html else {}
    doc_number = metadata.get("doc_number")
    doc_title = metadata.get("doc_title")

    if doc_title and doc_number:
        return slugify(f"{doc_title}_{doc_number}")

    if doc_title:
        return slugify(doc_title)

    url = record.get("canonical_url") or record.get("url") or ""
    path_slug = slugify(urlparse(url).path)
    url_hash = sha256_text(url)[:10]
    return slugify(f"{path_slug}_{url_hash}")


def read_jsonl(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break

    return records


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]], mode: str = "w") -> int:
    path = ensure_parent(path)
    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log_error(path: str | Path, record: Dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_markdown_noise(markdown: str) -> str:
    """
    Conservative cleanup for common website boilerplate.
    """
    markdown = normalize_text(markdown)
    noise_patterns = [
        r"(?im)^.*đăng nhập.*$",
        r"(?im)^.*đăng ký thành viên.*$",
        r"(?im)^.*vui lòng đăng nhập.*$",
        r"(?im)^.*vui lòng tài khoản.*$",
        r"(?im)^.*tiện ích dành cho tài khoản.*$",
        r"(?im)^.*xem chi tiết có hiệu lực.*$",
        r"(?im)^.*chia sẻ.*$",
        r"(?im)^.*tin liên quan.*$",
        r"(?im)^.*xem thêm.*$",
        r"(?im)^\|\s*tiện ích dành cho tài khoản.*\|?$",
        r"(?im)^\|\s*---\s*(\|\s*---\s*)+\|?$",
    ]
    for pattern in noise_patterns:
        markdown = re.sub(pattern, "", markdown)

    # Drop lines that repeat member-only prompts even if mixed with punctuation.
    cleaned_lines = []
    for line in markdown.splitlines():
        lowered = line.lower()
        if "tiện ích dành cho tài khoản" in lowered:
            continue
        if "vui lòng tài khoản" in lowered:
            continue
        if "xem chi tiết có hiệu lực" in lowered and "điều" not in lowered:
            continue
        cleaned_lines.append(line)

    markdown = "\n".join(cleaned_lines)
    markdown = normalize_text(markdown)

    # Trim website chrome before the first meaningful document section.
    content_markers = [
        "\n# ",
        "\n## TÓM TẮT",
        "\n## TOM TAT",
    ]
    starts = [markdown.find(marker) for marker in content_markers if markdown.find(marker) >= 0]
    if starts:
        markdown = markdown[min(starts) + 1 :]

    return normalize_text(markdown)


def get_markdown_text(markdown_obj: Any) -> str:
    if markdown_obj is None:
        return ""

    if hasattr(markdown_obj, "fit_markdown") and markdown_obj.fit_markdown:
        return str(markdown_obj.fit_markdown)

    if hasattr(markdown_obj, "raw_markdown") and markdown_obj.raw_markdown:
        return str(markdown_obj.raw_markdown)

    return str(markdown_obj)


def extract_label_value_from_text(text: str, labels: List[str]) -> Optional[str]:
    """
    Try multiple robust patterns to extract metadata from flattened page text.
    """
    for label in labels:
        # Pattern 1: label on same line: "Số hiệu: 59/2020/QH14"
        pattern = rf"(?im)^\s*{re.escape(label)}\s*[:：]?\s*(.+?)\s*$"
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip(" :：\t")
            if value and value.lower() != label.lower():
                return value

        # Pattern 2: label line followed by value line
        pattern = rf"(?im)^\s*{re.escape(label)}\s*$\n\s*(.+?)\s*$"
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip(" :：\t")
            if value and value.lower() != label.lower():
                return value

    return None


def extract_document_metadata(html: str, url: str) -> Dict[str, Any]:
    """
    Extract document-level metadata from detail page HTML.

    This is intentionally generic and can be refined for each provider later.
    """
    soup = BeautifulSoup(html or "", "html.parser")

    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""

    if not title:
        title_tag = soup.find("title")
        title = title_tag.get_text(" ", strip=True) if title_tag else ""

    # Remove script/style before text extraction
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    flat_text = soup.get_text("\n", strip=True)
    flat_text = normalize_text(flat_text)

    metadata = {
        "source_url": url,
        "doc_title": title,
        "doc_number": extract_label_value_from_text(flat_text, ["Số hiệu", "Số/Ký hiệu", "Số văn bản"]),
        "doc_type": extract_label_value_from_text(flat_text, ["Loại văn bản", "Loại VB"]),
        "issuing_body": extract_label_value_from_text(flat_text, ["Cơ quan ban hành", "Cơ quan ban hành/ Chức danh"]),
        "signer": extract_label_value_from_text(flat_text, ["Người ký", "Người ký ban hành"]),
        "issue_date": extract_label_value_from_text(flat_text, ["Ngày ban hành", "Ban hành"]),
        "effective_date": extract_label_value_from_text(flat_text, ["Ngày có hiệu lực", "Hiệu lực", "Ngày hiệu lực"]),
        "status": extract_label_value_from_text(flat_text, ["Tình trạng hiệu lực", "Hiệu lực hiện tại", "Tình trạng"]),
    }

    return metadata


def detect_access_restriction(html: str, markdown: str) -> Dict[str, Any]:
    text = f"{html or ''}\n{markdown or ''}".lower()

    signals = {
        "login_required": any(
            keyword in text
            for keyword in [
                "vui lòng đăng nhập",
                "bạn chưa đăng nhập",
                "đăng nhập thành viên",
                "login required",
            ]
        ),
        "paywall_or_member_content": any(
            keyword in text
            for keyword in [
                "tiện ích dành cho tài khoản",
                "xem chi tiết có hiệu lực",
                "tài khoản nâng cao",
                "tài khoản tiêu chuẩn",
                "tài khoản thành viên",
                "nội dung dành cho thành viên",
                "trả phí",
                "premium",
            ]
        ),
        "captcha_detected": any(
            keyword in text
            for keyword in [
                "captcha",
                "i'm not a robot",
                "g-recaptcha",
                "hcaptcha",
            ]
        ),
    }

    signals["has_restriction_signal"] = any(signals.values())
    return signals


async def crawl_detail_page(
    crawler: Any,
    record: Dict[str, Any],
    *,
    html_dir: Path,
    markdown_dir: Path,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    if CrawlerRunConfig is None or CacheMode is None:
        raise RuntimeError("crawl4ai is not installed. Run: pip install crawl4ai")

    url = record.get("url")
    if not url:
        raise ValueError("Document URL record is missing 'url'.")

    content_filter = PruningContentFilter(
        threshold=0.45,
        threshold_type="dynamic",
        min_word_threshold=10,
    )

    markdown_generator = DefaultMarkdownGenerator(
        content_filter=content_filter,
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        markdown_generator=markdown_generator,
        excluded_tags=["script", "style", "nav", "footer", "aside"],
        exclude_external_links=True,
        word_count_threshold=20,
        page_timeout=timeout_seconds * 1000,
    )

    result = await crawler.arun(url=url, config=run_config)

    if not result.success:
        return {
            **record,
            "success": False,
            "error_message": result.error_message,
            "crawl_time": utc_now(),
        }

    html = result.html or ""
    markdown = clean_markdown_noise(get_markdown_text(result.markdown))
    metadata = extract_document_metadata(html, url)
    doc_id = stable_doc_id(record, html=html)

    html_dir = ensure_dir(html_dir)
    markdown_dir = ensure_dir(markdown_dir)
    html_path = html_dir / f"{doc_id}.html"
    markdown_path = markdown_dir / f"{doc_id}.md"

    html_path.write_text(html, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")

    restriction = detect_access_restriction(html, markdown)

    return {
        **record,
        **metadata,
        "doc_id": doc_id,
        "success": True,
        "crawl_time": utc_now(),
        "raw_html_path": str(html_path),
        "markdown_path": str(markdown_path),
        "html_hash": sha256_text(html),
        "content_hash": sha256_text(markdown),
        "markdown_length": len(markdown),
        "access_restriction": restriction,
    }


def deduplicate_input_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    seen = set()

    for record in records:
        key = record.get("canonical_url") or record.get("url")
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(record)

    return output


async def run_crawl_documents(
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    html_dir: str | Path = DEFAULT_HTML_DIR,
    markdown_dir: str | Path = DEFAULT_MARKDOWN_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    error_log_path: str | Path = DEFAULT_ERROR_LOG_PATH,
    limit: Optional[int] = None,
    rate_limit_seconds: float = 2.0,
    timeout_seconds: int = 60,
    append_manifest: bool = False,
) -> Dict[str, Any]:
    if AsyncWebCrawler is None or BrowserConfig is None:
        raise RuntimeError(
            "crawl4ai is not installed. Run: pip install crawl4ai && crawl4ai-setup"
        )

    html_dir = ensure_dir(html_dir)
    markdown_dir = ensure_dir(markdown_dir)
    manifest_path = ensure_parent(manifest_path)
    report_path = ensure_parent(report_path)
    error_log_path = ensure_parent(error_log_path)

    input_records = read_jsonl(input_path, limit=limit)
    input_records = deduplicate_input_records(input_records)

    browser_config = BrowserConfig(
        headless=True,
        text_mode=True,
        java_script_enabled=True,
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )

    manifest_mode = "a" if append_manifest else "w"
    success_count = 0
    failed_count = 0
    restricted_count = 0
    domain_counts: Dict[str, int] = {}

    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest_file:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for idx, record in enumerate(input_records, start=1):
                try:
                    crawled = await crawl_detail_page(
                        crawler,
                        record,
                        html_dir=html_dir,
                        markdown_dir=markdown_dir,
                        timeout_seconds=timeout_seconds,
                    )

                    manifest_file.write(json.dumps(crawled, ensure_ascii=False) + "\n")

                    if crawled.get("success"):
                        success_count += 1
                        domain = crawled.get("domain", "unknown")
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1

                        restriction = crawled.get("access_restriction", {})
                        if restriction.get("has_restriction_signal"):
                            restricted_count += 1
                    else:
                        failed_count += 1
                        log_error(error_log_path, crawled)

                except Exception as exc:
                    failed_count += 1
                    error_record = {
                        **record,
                        "success": False,
                        "error_message": repr(exc),
                        "crawl_time": utc_now(),
                    }
                    manifest_file.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    log_error(error_log_path, error_record)

                await asyncio.sleep(rate_limit_seconds)

    report = {
        "task": "crawl_detail_pages",
        "created_at": utc_now(),
        "input_path": str(input_path),
        "manifest_path": str(manifest_path),
        "html_dir": str(html_dir),
        "markdown_dir": str(markdown_dir),
        "total_input_records": len(input_records),
        "success_count": success_count,
        "failed_count": failed_count,
        "restricted_signal_count": restricted_count,
        "domain_counts": domain_counts,
    }

    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--html-dir", default=str(DEFAULT_HTML_DIR))
    parser.add_argument("--markdown-dir", default=str(DEFAULT_MARKDOWN_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--error-log", default=str(DEFAULT_ERROR_LOG_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rate-limit-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--append-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    report = asyncio.run(
        run_crawl_documents(
            input_path=args.input,
            html_dir=args.html_dir,
            markdown_dir=args.markdown_dir,
            manifest_path=args.manifest,
            report_path=args.report,
            error_log_path=args.error_log,
            limit=args.limit,
            rate_limit_seconds=args.rate_limit_seconds,
            timeout_seconds=args.timeout_seconds,
            append_manifest=args.append_manifest,
        )
    )

    print("Crawl detail pages: DONE")
    print(f"Input records: {report['total_input_records']}")
    print(f"Success: {report['success_count']}")
    print(f"Failed: {report['failed_count']}")
    print(f"Restricted signals: {report['restricted_signal_count']}")
    print(f"Manifest: {report['manifest_path']}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
