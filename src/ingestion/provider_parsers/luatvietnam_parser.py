from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Optional

from bs4 import BeautifulSoup

from src.ingestion.common import normalize_text


DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _normalize_key(text: str) -> str:
    lowered = normalize_text(text or "").lower().replace("đ", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    plain = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


def _extract_date(value: Optional[str]) -> Optional[str]:
    text = normalize_text(value or "")
    match = DATE_RE.search(text)
    if match:
        return match.group(0)
    iso_match = ISO_DATE_RE.search(text)
    if not iso_match:
        return None
    year, month, day = iso_match.groups()
    return f"{day}/{month}/{year}"


def _clean_value(value: Optional[str]) -> Optional[str]:
    text = normalize_text(value or "")
    if not text:
        return None
    lowered = _normalize_key(text)
    blocked = {
        "dang nhap",
        "tai ve",
        "xem them",
        "tin lien quan",
        "luatvietnam",
        "tien ich danh cho tai khoan",
        "tieu chuan hoac nang cao",
    }
    if lowered in blocked:
        return None
    if any(signal in lowered for signal in ["vui long dang nhap", "tien ich danh cho tai khoan"]):
        return None
    return text


def _cell_text(cell) -> str:
    clone = BeautifulSoup(str(cell), "html.parser")
    for selector in [".tooltip-content-1", ".document-tip", "script", "style", "svg"]:
        for node in clone.select(selector):
            node.decompose()
    return normalize_text(clone.get_text(" ", strip=True))


def _extract_table_pairs(soup: BeautifulSoup) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for row in soup.select("table tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        for index in range(0, len(cells) - 1, 2):
            label_cell = cells[index]
            value_cell = cells[index + 1]
            label = label_cell.find("strong")
            raw_label = label.get_text(" ", strip=True) if label else _cell_text(label_cell)
            key = _normalize_key(raw_label)
            value = _clean_value(_cell_text(value_cell))
            if key and value:
                pairs[key] = value
    return pairs


def _extract_legislation_jsonld(soup: BeautifulSoup) -> Dict[str, str]:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        text = script.string or script.get_text()
        if not text.strip():
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        candidates = payload.get("@graph", []) if isinstance(payload, dict) else payload
        if not isinstance(candidates, list):
            candidates = [candidates]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if str(item.get("@type") or "").lower() != "legislation":
                continue
            return {
                "doc_title": _clean_value(item.get("name")),
                "doc_number": _clean_value(item.get("legislationIdentifier")),
                "doc_type": _clean_value(item.get("legislationType")),
                "issuing_body": _clean_value(item.get("legislationPassedBy")),
                "issue_date": _extract_date(item.get("datePublished")),
                "description": _clean_value(item.get("description")),
            }
    return {}


def _description_effective_date(texts: Iterable[Optional[str]]) -> Optional[str]:
    for text in texts:
        cleaned = normalize_text(text or "")
        if not cleaned:
            continue
        match = re.search(r"co hieu luc tu\s+(\d{1,2}/\d{1,2}/\d{4})", _normalize_key(cleaned))
        if match:
            return match.group(1)
        extracted = _extract_date(cleaned)
        if "hieu luc" in _normalize_key(cleaned) and extracted:
            return extracted
    return None


def parse_luatvietnam_metadata(
    html: str,
    markdown: str | None = None,
    url: str | None = None,
) -> Dict[str, object]:
    soup = BeautifulSoup(html or "", "html.parser")
    table_pairs = _extract_table_pairs(soup)
    jsonld = _extract_legislation_jsonld(soup)

    title = _clean_value((soup.find("h1") or {}).get_text(" ", strip=True) if soup.find("h1") else None)
    if not title:
        title = _clean_value(soup.title.get_text(" ", strip=True) if soup.title else None) or jsonld.get("doc_title")

    meta_description = None
    description_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    if description_tag:
        meta_description = _clean_value(description_tag.get("content"))

    issue_date = _extract_date(table_pairs.get("ngay ban hanh")) or jsonld.get("issue_date")
    effective_date = (
        _extract_date(table_pairs.get("ap dung"))
        or _description_effective_date([meta_description, jsonld.get("description"), markdown])
    )

    raw_status = (
        _clean_value(table_pairs.get("tinh trang hieu luc"))
        or _clean_value(table_pairs.get("hieu luc"))
        or _clean_value((soup.select_one(".item-status strong") or {}).get_text(" ", strip=True) if soup.select_one(".item-status strong") else None)
    )
    normalized_status = _normalize_key(raw_status or "")
    if normalized_status in {"da biet", "dang cap nhat"}:
        raw_status = None

    metadata = {
        "doc_title": title,
        "doc_number": _clean_value(table_pairs.get("so hieu")) or jsonld.get("doc_number"),
        "doc_type": _clean_value(table_pairs.get("loai van ban")) or jsonld.get("doc_type"),
        "issuing_body": _clean_value(table_pairs.get("co quan ban hanh")) or jsonld.get("issuing_body"),
        "signer": _clean_value(table_pairs.get("nguoi ky")),
        "issue_date": issue_date,
        "effective_date": effective_date,
        "status": raw_status,
        "confidence": {
            "doc_title": 0.95 if title else 0.0,
            "doc_number": 0.95 if _clean_value(table_pairs.get("so hieu")) else (0.85 if jsonld.get("doc_number") else 0.0),
            "doc_type": 0.95 if _clean_value(table_pairs.get("loai van ban")) else (0.85 if jsonld.get("doc_type") else 0.0),
            "issuing_body": 0.95 if _clean_value(table_pairs.get("co quan ban hanh")) else (0.85 if jsonld.get("issuing_body") else 0.0),
            "signer": 0.95 if _clean_value(table_pairs.get("nguoi ky")) else 0.0,
            "issue_date": 0.95 if _extract_date(table_pairs.get("ngay ban hanh")) else (0.85 if jsonld.get("issue_date") else 0.0),
            "effective_date": 0.95 if _extract_date(table_pairs.get("ap dung")) else (0.75 if effective_date else 0.0),
            "status": 0.8 if raw_status else 0.0,
            "provider": 1.0 if url and "luatvietnam.vn" in url else 0.9,
        },
    }
    return metadata


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Parse LuatVietnam document metadata from raw HTML.")
    parser.add_argument("--html", required=True)
    parser.add_argument("--url", default="https://luatvietnam.vn/test")
    args = parser.parse_args()
    html = Path(args.html).read_text(encoding="utf-8", errors="ignore")
    print(json.dumps(parse_luatvietnam_metadata(html, url=args.url), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
