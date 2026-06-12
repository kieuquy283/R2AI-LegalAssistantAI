from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/raw/hf_targeted_curated_smoke.jsonl")
DEFAULT_OUTPUT = Path("data/raw/hf_question_targeted_smoke.jsonl")

PRIORITY_DOC_NUMBERS = {
    "04/2017/QH14",
    "59/2020/QH14",
    "80/2021/ND-CP",
    "06/2019/TT-BKHDT",
    "07/2020/TT-BKHCN",
    "06/2022/TT-BKHDT",
    "38/2019/QH14",
    "78/2021/TT-BTC",
    "32/2025/TT-BTC",
    "22/2023/QH15",
    "23/2024/ND-CP",
    "65/2023/ND-CP",
    "274/2025/ND-CP",
}

TAX_PRIORITY_DOC_NUMBERS = {
    "78/2006/QH11",
    "85/2007/ND-CP",
    "60/2007/TT-BTC",
    "85/2007/TT-BTC",
    "157/2009/TT-BTC",
    "10/2009/TT-BTC",
    "51/2010/ND-CP",
    "74/2010/TT-BTC",
    "94/2010/TT-BTC",
    "102/2010/TTLT-BTC-NHNN",
}

PRIORITY_TITLE_PHRASES = [
    "ho tro doanh nghiep nho va vua",
    "luat doanh nghiep",
    "luat dau thau",
    "hoa don, chung tu",
    "hoa don dien tu",
    "luat quan ly thue",
    "dang ky thue",
    "khai thue",
    "quyet toan thue",
    "mien thue",
    "giam thue",
    "ngung su dung hoa don",
    "cuong che",
    "bao hiem xa hoi",
    "nhan hieu",
    "so huu tri tue",
]

SCENARIO_RULES = [
    {
        "name": "sme_support",
        "phrases": [
            "doanh nghiep nho va vua",
            "ho tro doanh nghiep nho va vua",
            "tu van vien",
            "uom tao",
            "khu lam viec chung",
        ],
        "weight": 5,
    },
    {
        "name": "procurement",
        "phrases": [
            "luat dau thau",
            "dau thau",
            "uu dai",
            "nha dau tu",
        ],
        "weight": 4,
    },
    {
        "name": "invoice_tax",
        "phrases": [
            "hoa don dien tu",
            "hoa don, chung tu",
            "hoa don",
            "co ma cua co quan thue",
            "ngung su dung hoa don",
            "cuong che",
            "quan ly thue",
            "dang ky thue",
            "khai thue",
            "quyet toan thue",
            "mien thue",
            "giam thue",
        ],
        "weight": 4,
    },
    {
        "name": "tax_management",
        "phrases": [
            "luat quan ly thue",
            "quan ly thue",
            "dang ky thue",
            "khai thue",
            "quyet toan thue",
            "thue gia tri gia tang",
            "thue thu nhap",
            "mien thue",
            "giam thue",
        ],
        "weight": 6,
    },
    {
        "name": "social_insurance",
        "phrases": [
            "bao hiem xa hoi",
            "cham dong",
            "tron dong",
        ],
        "weight": 3,
    },
    {
        "name": "ip",
        "phrases": [
            "nhan hieu",
            "xam pham",
            "so huu tri tue",
            "so huu cong nghiep",
        ],
        "weight": 3,
    },
]


def _normalize(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d").replace("Đ", "d")
    lowered = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")


def _parse_date(value: str) -> tuple[int, int, int]:
    text = str(value or "").strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return (dt.year, dt.month, dt.day)
        except ValueError:
            continue
    return (0, 0, 0)


def _scenario_score(text: str) -> tuple[int, list[str]]:
    matched: list[str] = []
    score = 0
    for rule in SCENARIO_RULES:
        hits = [phrase for phrase in rule["phrases"] if phrase in text]
        if not hits:
            continue
        matched.append(str(rule["name"]))
        score += len(hits) * int(rule["weight"])
    return score, matched


def _row_score(row: dict[str, Any]) -> tuple[int, list[str]]:
    doc_number = str(row.get("doc_number") or "").strip()
    title = str(row.get("doc_title") or "")
    content = str(row.get("content") or "")
    haystack = _normalize("\n".join([doc_number, title, content[:12000]]))

    score = 0
    matched_labels: list[str] = []

    if doc_number in PRIORITY_DOC_NUMBERS:
        score += 100
        matched_labels.append("priority_doc")
    if doc_number in TAX_PRIORITY_DOC_NUMBERS:
        score += 120
        matched_labels.append("tax_priority_doc")

    title_norm = _normalize(title)
    title_hits = [phrase for phrase in PRIORITY_TITLE_PHRASES if phrase in title_norm]
    if title_hits:
        score += len(title_hits) * 10
        matched_labels.extend(f"title:{phrase}" for phrase in title_hits)

    scenario_score, scenario_labels = _scenario_score(haystack)
    if scenario_score:
        score += scenario_score
        matched_labels.extend(scenario_labels)

    if "doanh nghiep nho va vua" in haystack and "dau thau" in haystack:
        score += 25
        matched_labels.append("sme_procurement_bridge")
    if "hoa don" in haystack and "quan ly thue" in haystack:
        score += 20
        matched_labels.append("invoice_tax_bridge")
    if "dang ky thue" in haystack or "khai thue" in haystack or "quyet toan thue" in haystack:
        score += 18
        matched_labels.append("tax_registration_bridge")
    if "ngung su dung hoa don" in haystack or "cuong che" in haystack:
        score += 15
        matched_labels.append("enforcement_invoice_bridge")

    year, _month, _day = _parse_date(str(row.get("issued_date") or ""))
    if year >= 2019:
        score += 3
    if year >= 2023:
        score += 2

    return score, matched_labels


def build_subset(input_path: Path, output_path: Path, *, top_k: int) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    seen_doc_numbers: set[str] = set()
    label_counter: Counter[str] = Counter()

    rows: list[tuple[int, tuple[int, int, int], dict[str, Any]]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            score, labels = _row_score(row)
            if score <= 0:
                continue
            row["_selection_score"] = score
            row["_selection_labels"] = labels
            rows.append((score, _parse_date(str(row.get("issued_date") or "")), row))

    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)

    for score, _issued_date, row in rows:
        doc_number = str(row.get("doc_number") or "").strip()
        if not doc_number or doc_number in seen_doc_numbers:
            continue
        seen_doc_numbers.add(doc_number)
        selected_rows.append(row)
        label_counter.update(row.get("_selection_labels") or [])
        if len(selected_rows) >= top_k:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "selected_count": len(selected_rows),
        "doc_numbers": [str(row.get("doc_number") or "") for row in selected_rows],
        "top_labels": dict(label_counter.most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny HF subset targeted to the current smoke questions.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=32)
    args = parser.parse_args()

    report = build_subset(Path(args.input), Path(args.output), top_k=args.top_k)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
