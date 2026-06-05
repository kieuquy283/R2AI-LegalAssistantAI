"""
Source registry loader and validator for R2AI Legal AI Assistant ingestion.

Usage:
    python -m src.ingestion.source_registry --sources data/sources/sources.yaml

This module:
- Loads sources.yaml
- Validates required fields
- Filters enabled sources
- Exposes helper functions for crawler pipeline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import yaml


REQUIRED_SOURCE_FIELDS = {
    "id",
    "name",
    "provider",
    "source_type",
    "domain",
    "url",
    "enabled",
    "priority",
    "crawl_frequency",
    "crawl_strategy",
    "extraction",
    "legal_scope",
    "compliance",
}


@dataclass(frozen=True)
class SourceConfig:
    id: str
    name: str
    provider: str
    source_type: str
    domain: str
    url: str
    enabled: bool
    priority: str
    crawl_frequency: str
    raw: Dict[str, Any]

    @property
    def is_search_page(self) -> bool:
        return self.source_type == "search_page"

    @property
    def should_collect_document_links(self) -> bool:
        return bool(
            self.raw.get("crawl_strategy", {}).get("collect_document_links", False)
        )

    @property
    def link_pattern(self) -> Optional[str]:
        return self.raw.get("crawl_strategy", {}).get("link_pattern")

    @property
    def max_pages(self) -> int:
        return int(
            self.raw.get("crawl_strategy", {})
            .get("pagination", {})
            .get("max_pages", 1)
        )


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source registry not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("sources.yaml must contain a YAML object at root level.")

    return data


def validate_registry(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    allowed_domains = set(data.get("allowed_domains", []))
    sources = data.get("sources", [])

    if not allowed_domains:
        errors.append("Missing or empty 'allowed_domains'.")

    if not isinstance(sources, list) or not sources:
        errors.append("Missing or empty 'sources' list.")
        return errors

    seen_ids = set()

    for idx, source in enumerate(sources):
        prefix = f"sources[{idx}]"

        if not isinstance(source, dict):
            errors.append(f"{prefix}: source must be an object.")
            continue

        missing = REQUIRED_SOURCE_FIELDS - set(source.keys())
        if missing:
            errors.append(f"{prefix}: missing required fields: {sorted(missing)}")

        source_id = source.get("id")
        if source_id in seen_ids:
            errors.append(f"{prefix}: duplicate id '{source_id}'.")
        if source_id:
            seen_ids.add(source_id)

        domain = source.get("domain")
        if domain and domain not in allowed_domains:
            errors.append(
                f"{prefix}: domain '{domain}' is not in allowed_domains."
            )

        related_domains = source.get("related_domains", [])
        if related_domains:
            for related in related_domains:
                if related not in allowed_domains:
                    errors.append(
                        f"{prefix}: related domain '{related}' is not in allowed_domains."
                    )

        url = source.get("url", "")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            errors.append(f"{prefix}: url must start with http:// or https://.")

        crawl_strategy = source.get("crawl_strategy", {})
        if crawl_strategy and not isinstance(crawl_strategy, dict):
            errors.append(f"{prefix}: crawl_strategy must be an object.")

        extraction = source.get("extraction", {})
        if extraction and not isinstance(extraction, dict):
            errors.append(f"{prefix}: extraction must be an object.")

        compliance = source.get("compliance", {})
        if compliance:
            if compliance.get("no_paywall_bypass") is not True:
                errors.append(
                    f"{prefix}: compliance.no_paywall_bypass should be true."
                )
            if compliance.get("no_login") is not True:
                errors.append(f"{prefix}: compliance.no_login should be true.")

    return errors


def get_sources(path: str | Path, enabled_only: bool = True) -> List[SourceConfig]:
    data = load_yaml(path)
    errors = validate_registry(data)

    if errors:
        joined = "\n".join(f"- {e}" for e in errors)
        raise ValueError(f"Invalid source registry:\n{joined}")

    sources: List[SourceConfig] = []

    for raw in data["sources"]:
        if enabled_only and not raw.get("enabled", False):
            continue

        sources.append(
            SourceConfig(
                id=raw["id"],
                name=raw["name"],
                provider=raw["provider"],
                source_type=raw["source_type"],
                domain=raw["domain"],
                url=raw["url"],
                enabled=raw["enabled"],
                priority=raw["priority"],
                crawl_frequency=raw["crawl_frequency"],
                raw=raw,
            )
        )

    return sources


def group_sources_by_domain(sources: List[SourceConfig]) -> Dict[str, List[SourceConfig]]:
    grouped: Dict[str, List[SourceConfig]] = {}

    for source in sources:
        grouped.setdefault(source.domain, []).append(source)

    return grouped


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        default="data/sources/sources.yaml",
        help="Path to source registry YAML file.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include disabled sources in output.",
    )
    args = parser.parse_args()

    sources = get_sources(
        args.sources,
        enabled_only=not args.include_disabled,
    )

    grouped = group_sources_by_domain(sources)

    print(f"Loaded {len(sources)} enabled source(s).")
    for domain, items in grouped.items():
        print(f"- {domain}: {len(items)} source(s)")
        for item in items:
            print(f"  • {item.id} | {item.name}")


if __name__ == "__main__":
    main()
