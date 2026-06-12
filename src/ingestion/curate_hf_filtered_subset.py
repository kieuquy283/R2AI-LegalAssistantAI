from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_INPUT = Path("data/raw/hf_targeted_curated_smoke.jsonl")
DEFAULT_OUTPUT = Path("data/raw/hf_targeted_eval_smoke.jsonl")

GROUP_LIMITS = {
    "business_sme": 80,
    "tax_invoice_accounting": 120,
    "labor_bhxh_union": 120,
    "intellectual_property": 80,
    "commerce_procurement_customs_logistics": 60,
}

PREFERRED_ISSUERS = [
    "Quốc hội",
    "Chính phủ",
    "Thủ tướng Chính phủ",
    "Bộ Tài chính",
    "Bộ Kế hoạch và Đầu tư",
    "Bộ Lao động - Thương binh và Xã hội",
    "Bộ Khoa học và Công nghệ",
    "Bảo hiểm Xã hội Việt Nam",
    "Tổng cục Thuế",
    "Tổng cục Hải quan",
    "Bộ Công Thương",
]


def _parse_date(value: str) -> tuple[int, int, int]:
    text = str(value or "").strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return (dt.year, dt.month, dt.day)
        except ValueError:
            continue
    return (0, 0, 0)


def _issuer_score(value: str) -> int:
    issuer = str(value or "").strip()
    return 1 if issuer in PREFERRED_ISSUERS else 0


def curate_subset(input_path: Path, output_path: Path, limits: dict[str, int] | None = None) -> dict[str, int]:
    limits = limits or GROUP_LIMITS
    rows_by_group: dict[str, list[dict]] = defaultdict(list)
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_group[str(row.get("matched_group") or "")].append(row)

    selected: list[dict] = []
    counts: dict[str, int] = {}
    for group, rows in rows_by_group.items():
        rows.sort(
            key=lambda row: (
                _issuer_score(row.get("issuer")),
                _parse_date(row.get("issued_date")),
                len(str(row.get("doc_title") or "")),
            ),
            reverse=True,
        )
        limit = limits.get(group, 50)
        chosen = rows[:limit]
        selected.extend(chosen)
        counts[group] = len(chosen)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts["total"] = len(selected)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate a smaller eval smoke subset from filtered HF records.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--business-limit", type=int, default=GROUP_LIMITS["business_sme"])
    parser.add_argument("--tax-limit", type=int, default=GROUP_LIMITS["tax_invoice_accounting"])
    parser.add_argument("--labor-limit", type=int, default=GROUP_LIMITS["labor_bhxh_union"])
    parser.add_argument("--ip-limit", type=int, default=GROUP_LIMITS["intellectual_property"])
    parser.add_argument("--commerce-limit", type=int, default=GROUP_LIMITS["commerce_procurement_customs_logistics"])
    args = parser.parse_args()
    limits = {
        "business_sme": args.business_limit,
        "tax_invoice_accounting": args.tax_limit,
        "labor_bhxh_union": args.labor_limit,
        "intellectual_property": args.ip_limit,
        "commerce_procurement_customs_logistics": args.commerce_limit,
    }
    counts = curate_subset(Path(args.input), Path(args.output), limits=limits)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
