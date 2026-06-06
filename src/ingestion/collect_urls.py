"""
Collect document detail URLs from configured legal source search pages.

This module is Task 2 in the ingestion pipeline.

Input:
    data/sources/sources.yaml

Output:
    data/raw/document_urls.jsonl
    data/raw/document_urls_report.json
    data/logs/collect_urls_errors.jsonl

Usage:
    python -m src.ingestion.collect_urls \
        --sources data/sources/sources.yaml \
        --output data/raw/document_urls.jsonl \
        --limit-sources 2

Notes:
    - This module only collects public detail URLs from search/list pages.
    - It does not bypass login, paywall, CAPTCHA, or access restrictions.
    - It respects configured rate limit and max_pages.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", str(Path.cwd() / "data"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path.cwd() / "data" / "playwright-browsers"))

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
except Exception:  # pragma: no cover - handled at runtime
    AsyncWebCrawler = None
    BrowserConfig = None
    CrawlerRunConfig = None
    CacheMode = None

from .source_registry import SourceConfig, get_sources


DEFAULT_OUTPUT_PATH = Path("data/raw/document_urls.jsonl")
DEFAULT_REPORT_PATH = Path("data/raw/document_urls_report.json")
DEFAULT_ERROR_LOG_PATH = Path("data/logs/collect_urls_errors.jsonl")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_url(url: str) -> str:
    """
    Normalize URL for deduplication.

    Keeps query string because some legal sites use query params to identify docs,
    but removes fragments and normalizes trailing spaces.
    """
    url = url.strip()
    parsed = urlparse(url)
    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)


def canonicalize_detail_url(url: str) -> str:
    """
    Canonical detail URL key for deduplication.

    Removes common tracking parameters while preserving meaningful parameters.
    """
    parsed = urlparse(normalize_url(url))
    tracking_prefixes = ("utm_",)
    tracking_keys = {"fbclid", "gclid", "yclid"}

    query_pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key in tracking_keys or key.startswith(tracking_prefixes):
            continue
        query_pairs.append((key, value))

    parsed = parsed._replace(query=urlencode(query_pairs, doseq=True), fragment="")
    return urlunparse(parsed)


def is_same_host_or_relative(base_url: str, candidate_url: str) -> bool:
    base_host = urlparse(base_url).netloc
    candidate_host = urlparse(candidate_url).netloc
    return not candidate_host or candidate_host == base_host


def build_paginated_urls(source: SourceConfig) -> List[str]:
    """
    Create candidate search page URLs.

    Strategy:
    - Always include original source.url.
    - If pagination.strategy is manual_url_param, use param_name and page_start/page_end.
    - If detect_or_manual, keep only original URL for now; downstream crawler can be
      improved to detect next links or interactive pagination.
    """
    raw = source.raw
    pagination = raw.get("crawl_strategy", {}).get("pagination", {})
    enabled = bool(pagination.get("enabled", False))

    if not enabled:
        return [source.url]

    strategy = pagination.get("strategy", "detect_or_manual")
    max_pages = int(pagination.get("max_pages", 1))

    if strategy != "manual_url_param":
        return [source.url]

    param_name = pagination.get("param_name", "page")
    page_start = int(pagination.get("page_start", 1))
    page_end = int(pagination.get("page_end", max_pages))

    parsed = urlparse(source.url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))

    urls = []
    for page in range(page_start, page_end + 1):
        query[param_name] = str(page)
        new_url = urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
        urls.append(new_url)

    return urls


def extract_links_from_html(
    html: str,
    source: SourceConfig,
    page_url: str,
) -> List[Dict[str, Any]]:
    """
    Extract candidate detail document links from a search/list page.

    The primary filter is source.raw.crawl_strategy.link_pattern, e.g. "-d1.html".
    """
    soup = BeautifulSoup(html or "", "html.parser")

    link_pattern = source.link_pattern
    base_url = page_url or source.url

    records: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        absolute_url = normalize_url(urljoin(base_url, href))

        if not is_same_host_or_relative(source.url, absolute_url):
            continue

        if link_pattern and link_pattern not in absolute_url:
            continue

        text = a.get_text(" ", strip=True)
        canonical_url = canonicalize_detail_url(absolute_url)

        if canonical_url in seen:
            continue

        seen.add(canonical_url)

        records.append(
            {
                "url": absolute_url,
                "canonical_url": canonical_url,
                "anchor_text": text,
                "source_id": source.id,
                "source_name": source.name,
                "provider": source.provider,
                "domain": source.domain,
                "source_type": source.source_type,
                "collected_from": page_url,
                "link_pattern": link_pattern,
                "collected_at": utc_now(),
            }
        )

    return records


def extract_next_page_links(html: str, source: SourceConfig, page_url: str) -> List[str]:
    """
    Conservative next-page detection.

    It detects anchor text usually used for pagination. This is intentionally
    conservative to avoid crawling unrelated pages.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    next_text_patterns = [
        "tiếp",
        "sau",
        "next",
        ">",
        "›",
        "»",
    ]

    page_urls: List[str] = []
    source_host = urlparse(source.url).netloc

    for a in soup.select("a[href]"):
        text = a.get_text(" ", strip=True).lower()
        href = (a.get("href") or "").strip()
        if not href:
            continue

        if text not in next_text_patterns and not any(k in text for k in ["tiếp", "next"]):
            continue

        absolute_url = normalize_url(urljoin(page_url, href))
        if urlparse(absolute_url).netloc != source_host:
            continue

        page_urls.append(absolute_url)

    # Deduplicate while preserving order
    output = []
    seen = set()
    for url in page_urls:
        if url not in seen:
            seen.add(url)
            output.append(url)

    return output


async def crawl_html_with_crawl4ai(
    crawler: Any,
    url: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    if CrawlerRunConfig is None or CacheMode is None:
        raise RuntimeError("crawl4ai is not installed. Run: pip install crawl4ai")

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        page_timeout=timeout_seconds * 1000,
    )

    result = await crawler.arun(url=url, config=run_config)

    return {
        "success": bool(result.success),
        "html": result.html or "",
        "error_message": result.error_message if not result.success else None,
        "final_url": getattr(result, "url", url) or url,
    }


async def collect_for_source(
    crawler: Any,
    source: SourceConfig,
    *,
    max_pages_override: Optional[int] = None,
    timeout_seconds: int = 60,
    rate_limit_seconds: float = 2.0,
    error_log_path: Path = DEFAULT_ERROR_LOG_PATH,
) -> Dict[str, Any]:
    """
    Collect document URLs for a single source.
    """
    if not source.is_search_page or not source.should_collect_document_links:
        return {
            "source_id": source.id,
            "skipped": True,
            "reason": "source is not a collectable search page",
            "records": [],
            "pages_crawled": 0,
            "errors": 0,
        }

    candidate_pages = build_paginated_urls(source)
    pagination = source.raw.get("crawl_strategy", {}).get("pagination", {})
    max_pages = int(max_pages_override or pagination.get("max_pages", len(candidate_pages) or 1))

    queue: List[str] = candidate_pages[:max_pages]
    visited_pages: Set[str] = set()
    records: List[Dict[str, Any]] = []
    errors = 0

    while queue and len(visited_pages) < max_pages:
        page_url = queue.pop(0)
        page_url = normalize_url(page_url)

        if page_url in visited_pages:
            continue

        visited_pages.add(page_url)

        try:
            crawl_result = await crawl_html_with_crawl4ai(
                crawler,
                page_url,
                timeout_seconds=timeout_seconds,
            )

            if not crawl_result["success"]:
                errors += 1
                log_error(
                    error_log_path,
                    {
                        "source_id": source.id,
                        "page_url": page_url,
                        "error_type": "crawl_failed",
                        "error_message": crawl_result.get("error_message"),
                        "timestamp": utc_now(),
                    },
                )
                await asyncio.sleep(rate_limit_seconds)
                continue

            html = crawl_result["html"]
            final_url = crawl_result.get("final_url") or page_url

            page_records = extract_links_from_html(
                html=html,
                source=source,
                page_url=final_url,
            )
            records.extend(page_records)

            # Conservative next page discovery only for detect_or_manual.
            strategy = pagination.get("strategy", "detect_or_manual")
            if strategy == "detect_or_manual":
                for next_url in extract_next_page_links(html, source, final_url):
                    if next_url not in visited_pages and next_url not in queue:
                        if len(visited_pages) + len(queue) < max_pages:
                            queue.append(next_url)

        except Exception as exc:
            errors += 1
            log_error(
                error_log_path,
                {
                    "source_id": source.id,
                    "page_url": page_url,
                    "error_type": "exception",
                    "error_message": repr(exc),
                    "timestamp": utc_now(),
                },
            )

        await asyncio.sleep(rate_limit_seconds)

    # Deduplicate inside source
    records = deduplicate_records(records)

    return {
        "source_id": source.id,
        "source_name": source.name,
        "domain": source.domain,
        "skipped": False,
        "records": records,
        "pages_crawled": len(visited_pages),
        "errors": errors,
    }


def deduplicate_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    seen = set()

    for record in records:
        key = record.get("canonical_url") or canonicalize_detail_url(record["url"])
        if key in seen:
            continue
        seen.add(key)
        record["url_hash"] = sha256_text(key)
        output.append(record)

    return output


def log_error(path: str | Path, record: Dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    path = ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_report(path: str | Path, report: Dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


async def run_collect_urls(
    *,
    sources_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    error_log_path: str | Path = DEFAULT_ERROR_LOG_PATH,
    limit_sources: Optional[int] = None,
    limit_pages: Optional[int] = None,
    include_disabled: bool = False,
    domain: Optional[str] = None,
    rate_limit_seconds: Optional[float] = None,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    if AsyncWebCrawler is None or BrowserConfig is None:
        raise RuntimeError(
            "crawl4ai is not installed. Run: pip install crawl4ai && crawl4ai-setup"
        )

    sources = get_sources(sources_path, enabled_only=not include_disabled)

    if domain:
        sources = [s for s in sources if s.domain == domain]

    if limit_sources:
        sources = sources[:limit_sources]

    default_rate_limit = 2.0
    default_timeout = 60

    all_records: List[Dict[str, Any]] = []
    source_reports: List[Dict[str, Any]] = []

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

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for source in sources:
            source_rate_limit = float(
                rate_limit_seconds
                if rate_limit_seconds is not None
                else source.raw.get("default_rate_limit_seconds", default_rate_limit)
            )

            source_timeout = int(
                timeout_seconds
                if timeout_seconds is not None
                else source.raw.get("default_timeout_seconds", default_timeout)
            )

            result = await collect_for_source(
                crawler,
                source,
                max_pages_override=limit_pages,
                timeout_seconds=source_timeout,
                rate_limit_seconds=source_rate_limit,
                error_log_path=Path(error_log_path),
            )

            source_records = result.pop("records", [])
            all_records.extend(source_records)
            source_reports.append(
                {
                    **result,
                    "records_found": len(source_records),
                }
            )

    all_records = deduplicate_records(all_records)
    total_written = write_jsonl(output_path, all_records)

    report = {
        "task": "collect_document_urls",
        "created_at": utc_now(),
        "sources_path": str(sources_path),
        "output_path": str(output_path),
        "total_sources": len(sources),
        "total_unique_document_urls": total_written,
        "source_reports": source_reports,
    }

    write_report(report_path, report)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="data/sources/sources.yaml")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--error-log", default=str(DEFAULT_ERROR_LOG_PATH))
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--limit-pages", type=int, default=None)
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--rate-limit-seconds", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    report = asyncio.run(
        run_collect_urls(
            sources_path=args.sources,
            output_path=args.output,
            report_path=args.report,
            error_log_path=args.error_log,
            limit_sources=args.limit_sources,
            limit_pages=args.limit_pages,
            include_disabled=args.include_disabled,
            domain=args.domain,
            rate_limit_seconds=args.rate_limit_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    )

    print("Collect document URLs: DONE")
    print(f"Total sources: {report['total_sources']}")
    print(f"Total unique document URLs: {report['total_unique_document_urls']}")
    print(f"Output: {report['output_path']}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
