from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from huggingface_hub import HfFileSystem

from src.ingestion.common import ensure_parent, normalize_text, sha256_text, slugify_vi, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "th1nhng0/vietnamese-legal-documents"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "hf_targeted_curated_smoke.jsonl"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "logs" / "debug" / "hf_targeted_curated_smoke_report.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "logs" / "debug" / "hf_targeted_curated_smoke_report.md"

TARGET_RULES = [
    {
        "matched_group": "business_sme",
        "domain": "sme_support",
        "phrases": [
            "ho tro doanh nghiep nho va vua",
            "doanh nghiep nho va vua",
            "luat doanh nghiep",
            "dang ky doanh nghiep",
            "dang ky kinh doanh",
            "ho kinh doanh",
            "quy bao lanh tin dung cho doanh nghiep nho va vua",
            "quy phat trien doanh nghiep nho va vua",
            "co so uom tao doanh nghiep nho va vua",
            "khu lam viec chung ho tro doanh nghiep nho va vua",
        ],
    },
    {
        "matched_group": "tax_invoice_accounting",
        "domain": "tax",
        "phrases": [
            "quan ly thue",
            "hoa don",
            "chung tu",
            "ke toan",
            "bao cao tai chinh",
            "dang ky thue",
        ],
    },
    {
        "matched_group": "labor_bhxh_union",
        "domain": "labor",
        "phrases": [
            "bao hiem xa hoi",
            "bo luat lao dong",
            "hop dong lao dong",
            "luat cong doan",
            "luat viec lam",
            "an toan ve sinh lao dong",
        ],
    },
    {
        "matched_group": "intellectual_property",
        "domain": "intellectual_property",
        "phrases": [
            "so huu tri tue",
            "nhan hieu",
            "quyen tac gia",
            "quyen lien quan",
            "so huu cong nghiep",
        ],
    },
    {
        "matched_group": "commerce_procurement_customs_logistics",
        "domain": "commerce",
        "phrases": [
            "dau thau",
            "thuong mai dien tu",
            "dich vu logistics",
            "hai quan",
            "xuat khau",
            "nhap khau",
            "xuat nhap khau",
            "bao ve nguoi tieu dung",
        ],
    },
]

PREFERRED_DOC_TYPES = {
    "luat",
    "bo luat",
    "nghi dinh",
    "thong tu",
    "thong tu lien tich",
    "nghi quyet",
    "quyet dinh",
    "chi thi",
}

PREFERRED_ISSUERS = [
    "quoc hoi",
    "uy ban thuong vu quoc hoi",
    "chinh phu",
    "thu tuong chinh phu",
    "bo tai chinh",
    "bo ke hoach va dau tu",
    "bo lao dong",
    "bo lao dong thuong binh va xa hoi",
    "bao hiem xa hoi viet nam",
    "bo khoa hoc va cong nghe",
    "tong cuc thue",
    "tong cuc hai quan",
    "bo cong thuong",
]


def _dataset_repo_prefix(dataset_name: str) -> str:
    return f"datasets/{dataset_name}/data"


def _normalize(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d").replace("Đ", "d")
    lowered = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")


def _clean_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return normalize_text(soup.get_text("\n", strip=True))


def _title_matches(title: str) -> tuple[str, str, list[str]]:
    title_norm = _normalize(title)
    for rule in TARGET_RULES:
        matched = [phrase for phrase in rule["phrases"] if phrase in title_norm]
        if matched:
            return str(rule["matched_group"]), str(rule["domain"]), matched
    return "", "", []


def _preferred_context(doc_type: str, issuer: str) -> bool:
    doc_type_norm = _normalize(doc_type)
    issuer_norm = _normalize(issuer)
    type_ok = any(doc_type_norm.startswith(item) for item in PREFERRED_DOC_TYPES)
    issuer_ok = any(item in issuer_norm for item in PREFERRED_ISSUERS)
    return type_ok and issuer_ok


def _iter_parquet_rows(dataset_name: str, parquet_name: str, columns: list[str] | None = None):
    fs = HfFileSystem()
    parquet_path = f"{_dataset_repo_prefix(dataset_name)}/{parquet_name}.parquet"
    with fs.open(parquet_path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        for row_group_index in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group_index, columns=columns)
            for row in table.to_pylist():
                if isinstance(row, dict):
                    yield row


def build_targeted_subset(dataset_name: str, output_path: Path) -> dict[str, Any]:
    selected_meta: dict[str, dict[str, Any]] = {}
    group_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()

    for row in _iter_parquet_rows(
        dataset_name,
        "metadata",
        columns=["id", "title", "so_ky_hieu", "loai_van_ban", "co_quan_ban_hanh", "ngay_ban_hanh", "ngay_co_hieu_luc"],
    ):
        source_id = str(row.get("id") or "").strip()
        title = str(row.get("title") or "").strip()
        doc_number = str(row.get("so_ky_hieu") or "").strip()
        doc_type = str(row.get("loai_van_ban") or "").strip()
        issuer = str(row.get("co_quan_ban_hanh") or "").strip()
        matched_group, domain, matched_keywords = _title_matches(title)
        if not matched_group:
            continue
        if not _preferred_context(doc_type, issuer):
            continue
        doc_slug = slugify_vi(f"{title}_{doc_number}") if title or doc_number else source_id
        selected_meta[source_id] = {
            "source_id": source_id,
            "doc_id": source_id,
            "doc_slug": doc_slug,
            "doc_title": title,
            "doc_type": doc_type,
            "doc_number": doc_number,
            "issuer": issuer,
            "issued_date": str(row.get("ngay_ban_hanh") or "").strip(),
            "effective_date": str(row.get("ngay_co_hieu_luc") or "").strip(),
            "domain": domain,
            "candidate_domains": [domain],
            "matched_group": matched_group,
            "matched_keywords": matched_keywords,
            "priority": 1,
            "source_url": "",
        }
        group_counter.update([matched_group])
        domain_counter.update([domain])

    ensure_parent(output_path)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in _iter_parquet_rows(dataset_name, "content", columns=["id", "content_html"]):
            source_id = str(row.get("id") or "").strip()
            metadata = selected_meta.get(source_id)
            if not metadata:
                continue
            content = _clean_html(str(row.get("content_html") or ""))
            if not content:
                continue
            payload = dict(metadata)
            payload["source_dataset"] = dataset_name
            payload["content"] = content
            payload["content_hash"] = sha256_text(content)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    report = {
        "dataset": dataset_name,
        "selected_metadata_records": len(selected_meta),
        "written_content_records": written,
        "count_by_group": dict(group_counter.most_common()),
        "count_by_domain": dict(domain_counter.most_common()),
        "output_path": str(output_path),
    }
    return report


def _render_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# HF Targeted Curated Smoke Report",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Selected metadata records: `{report['selected_metadata_records']}`",
        f"- Written content records: `{report['written_content_records']}`",
        "",
        "## Count By Group",
        "",
    ]
    for key, value in report["count_by_group"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Count By Domain", ""])
    for key, value in report["count_by_domain"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a targeted HF legal smoke subset using title metadata only.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    args = parser.parse_args()

    report = build_targeted_subset(args.dataset, Path(args.output))
    write_json(args.report_json, report)
    Path(args.report_md).write_text(_render_report_md(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
