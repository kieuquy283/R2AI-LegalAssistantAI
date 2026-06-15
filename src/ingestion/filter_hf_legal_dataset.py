from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from src.ingestion.common import ensure_parent, normalize_text, sha256_text, slugify_vi, write_json
from src.ingestion.hf_legal_filter_rules import evaluate_hf_legal_filter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "th1nhng0/vietnamese-legal-documents"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "hf_filtered_business_sme.jsonl"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "logs" / "debug" / "hf_filter_report.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "logs" / "debug" / "hf_filter_report.md"


def _dataset_repo_prefix(dataset_name: str) -> str:
    return f"datasets/{dataset_name}/data"


def _clean_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return normalize_text(soup.get_text("\n", strip=True))


def _iter_parquet_rows(dataset_name: str, parquet_name: str) -> Iterable[dict[str, Any]]:
    fs = HfFileSystem()
    parquet_path = f"{_dataset_repo_prefix(dataset_name)}/{parquet_name}.parquet"
    with fs.open(parquet_path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        for row_group_index in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group_index)
            for row in table.to_pylist():
                if isinstance(row, dict):
                    yield row


def _normalize_metadata_row(row: dict[str, Any]) -> dict[str, Any]:
    source_id = str(row.get("id") or "").strip()
    doc_title = str(row.get("title") or "").strip()
    doc_number = str(row.get("so_ky_hieu") or "").strip()
    doc_type = str(row.get("loai_van_ban") or "").strip()
    issuer = str(row.get("co_quan_ban_hanh") or "").strip()
    issued_date = str(row.get("ngay_ban_hanh") or "").strip()
    effective_date = str(row.get("ngay_co_hieu_luc") or "").strip()
    source_hint = str(row.get("nguon_thu_thap") or "").strip()
    doc_slug = slugify_vi(f"{doc_title}_{doc_number}") if doc_title or doc_number else source_id
    return {
        "source_id": source_id,
        "doc_id": source_id,
        "doc_slug": doc_slug,
        "doc_title": doc_title,
        "doc_type": doc_type,
        "doc_number": doc_number,
        "issuer": issuer,
        "issued_date": issued_date,
        "effective_date": effective_date,
        "source_url": "",
        "source_hint": source_hint,
        "raw_metadata": row,
    }


def _join_metadata_and_content(dataset_name: str, limit: int | None = None) -> Iterable[dict[str, Any]]:
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for row in _iter_parquet_rows(dataset_name, "metadata"):
        normalized = _normalize_metadata_row(row)
        if normalized["source_id"]:
            metadata_by_id[normalized["source_id"]] = normalized

    yielded = 0
    for row in _iter_parquet_rows(dataset_name, "content"):
        source_id = str(row.get("id") or "").strip()
        metadata = metadata_by_id.get(source_id)
        if not metadata:
            continue
        content = _clean_html(str(row.get("content_html") or ""))
        if not content:
            continue
        merged = dict(metadata)
        merged["content"] = content
        merged["content_hash"] = sha256_text(content)
        yield merged
        yielded += 1
        if limit is not None and yielded >= limit:
            break


def _dedupe_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_url") or "").strip(),
        str(row.get("doc_title") or "").strip(),
        str(row.get("doc_number") or "").strip() or str(row.get("content_hash") or "").strip(),
    )


def _render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# HF Filter Report",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Total scanned records: `{report['total_scanned_records']}`",
        f"- Total matched records: `{report['total_matched_records']}`",
        f"- Total deduplicated records: `{report['total_deduplicated_records']}`",
        "",
        "## Count By Domain",
        "",
    ]
    for key, value in report["count_by_domain"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Count By Matched Group", ""])
    for key, value in report["count_by_matched_group"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Top Matched Keywords", ""])
    for key, value in report["top_matched_keywords"]:
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Top Document Titles", ""])
    for key, value in report["top_document_titles"]:
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Sample Matched Docs Per Group", ""])
    for group_name, samples in report["sample_matched_docs_per_group"].items():
        lines.append(f"### `{group_name}`")
        lines.append("")
        for sample in samples:
            lines.append("```json")
            lines.append(json.dumps(sample, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_filter(dataset_name: str, output_path: Path, limit: int | None = None) -> dict[str, Any]:
    scanned = 0
    matched = 0
    deduped = 0
    seen_keys: set[tuple[str, str, str]] = set()
    
    # INCREMENTAL UPDATE: Load existing hashes to skip already processed data
    existing_hashes = set()
    if output_path.exists():
        print(f"[INFO] Found existing file: {output_path}. Loading existing hashes to skip duplicates...")
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_data = json.loads(line)
                    existing_hashes.add(existing_data.get("content_hash", ""))
                except json.JSONDecodeError:
                    continue
        print(f"[INFO] Loaded {len(existing_hashes)} existing records. Will skip them.")

    domain_counter: Counter[str] = Counter()
    group_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    title_counter: Counter[str] = Counter()
    sample_docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    iterator = _join_metadata_and_content(dataset_name=dataset_name, limit=limit)
    ensure_parent(output_path)
    
    # Use "a" (append) mode to add new records without deleting old ones
    with output_path.open("a", encoding="utf-8") as handle:
        for row in tqdm(iterator, desc="filter_hf_legal_dataset", unit="doc"):
            scanned += 1
            
            # Skip if already processed in previous runs
            content_hash = str(row.get("content_hash") or sha256_text(str(row.get("content") or ""))).strip()
            if content_hash in existing_hashes:
                continue

            filter_result = evaluate_hf_legal_filter(row)
            if not filter_result["include"]:
                continue
            matched += 1

            normalized_row = {
                "source_dataset": dataset_name,
                "source_id": row["source_id"],
                "doc_id": row["doc_id"],
                "doc_title": row["doc_title"],
                "doc_type": row["doc_type"],
                "doc_number": row["doc_number"],
                "issuer": row["issuer"],
                "issued_date": row["issued_date"],
                "effective_date": row["effective_date"],
                "domain": filter_result["domain"],
                "candidate_domains": filter_result["candidate_domains"],
                "matched_group": filter_result["matched_group"],
                "matched_keywords": filter_result["matched_keywords"],
                "priority": filter_result["priority"],
                "source_url": row["source_url"],
                "content": row["content"],
                "content_hash": content_hash,
            }

            dedupe_key = _dedupe_key(normalized_row)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            deduped += 1

            handle.write(json.dumps(normalized_row, ensure_ascii=False) + "\n")
            domain_counter.update(normalized_row["candidate_domains"])
            group_counter.update([normalized_row["matched_group"]])
            keyword_counter.update(normalized_row["matched_keywords"])
            title_counter.update([normalized_row["doc_title"]])
            if len(sample_docs[normalized_row["matched_group"]]) < 3:
                sample_docs[normalized_row["matched_group"]].append(
                    {
                        "doc_title": normalized_row["doc_title"],
                        "doc_number": normalized_row["doc_number"],
                        "domain": normalized_row["domain"],
                        "matched_keywords": normalized_row["matched_keywords"][:10],
                    }
                )
    report = {
        "dataset": dataset_name,
        "total_scanned_records": scanned,
        "total_matched_records": matched,
        "total_deduplicated_records": deduped,
        "count_by_domain": dict(domain_counter.most_common()),
        "count_by_matched_group": dict(group_counter.most_common()),
        "top_matched_keywords": keyword_counter.most_common(20),
        "top_document_titles": title_counter.most_common(20),
        "sample_matched_docs_per_group": dict(sample_docs),
        "output_path": str(output_path),
    }
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Filter the Hugging Face legal dataset into the legal RAG schema.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--streaming", action="store_true", help="Accepted for CLI compatibility.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    args = parser.parse_args()

    output_path = Path(args.output)
    report_json_path = Path(args.report_json)
    report_md_path = Path(args.report_md)
    ensure_parent(output_path)
    ensure_parent(report_json_path)
    ensure_parent(report_md_path)

    report = run_filter(dataset_name=args.dataset, output_path=output_path, limit=args.limit)
    write_json(report_json_path, report)
    report_md_path.write_text(_render_markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
