from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import DatasetDict, IterableDatasetDict, get_dataset_config_names, load_dataset, load_dataset_builder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = PROJECT_ROOT / "logs" / "debug" / "hf_dataset_schema_report.json"
REPORT_MD_PATH = PROJECT_ROOT / "logs" / "debug" / "hf_dataset_schema_report.md"

FIELD_CANDIDATES = {
    "id": ["id", "source_id", "doc_id", "uuid", "record_id"],
    "title": ["title", "doc_title", "name", "document_title", "document_name"],
    "content": ["text", "content", "full_text", "body", "document_text", "raw_text"],
    "doc_type": ["doc_type", "type", "document_type", "kind"],
    "document_number": ["document_number", "doc_number", "code", "number", "symbol", "so_hieu", "so_ky_hieu"],
    "issuer": ["issuer", "agency", "issuing_body", "organization", "co_quan_ban_hanh"],
    "issued_date": ["issued_date", "date", "issue_date", "publication_date", "ngay_ban_hanh"],
    "effective_date": ["effective_date", "effective_from", "ngay_hieu_luc"],
    "source_url": ["source_url", "url", "link", "source", "canonical_url"],
}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sample_rows(split_dataset: Any, sample_count: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if sample_count <= 0:
        return samples

    if hasattr(split_dataset, "take"):
        iterator = split_dataset.take(sample_count)
        for row in iterator:
            if isinstance(row, dict):
                samples.append(row)
        return samples

    limit = min(sample_count, len(split_dataset))
    for index in range(limit):
        row = split_dataset[index]
        if isinstance(row, dict):
            samples.append(row)
    return samples


def _row_count(split_dataset: Any) -> int | None:
    if hasattr(split_dataset, "num_rows"):
        return int(split_dataset.num_rows)
    try:
        return len(split_dataset)
    except Exception:
        return None


def _column_names(split_dataset: Any, sample_rows: list[dict[str, Any]]) -> list[str]:
    if hasattr(split_dataset, "column_names"):
        columns = getattr(split_dataset, "column_names")
        if columns:
            return list(columns)
    if sample_rows:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in sample_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        return ordered
    return []


def _detect_fields(column_names: Iterable[str]) -> dict[str, dict[str, Any]]:
    lowered_map = {str(column).lower(): str(column) for column in column_names}
    detected: dict[str, dict[str, Any]] = {}
    for logical_name, candidates in FIELD_CANDIDATES.items():
        matched = next((lowered_map[candidate.lower()] for candidate in candidates if candidate.lower() in lowered_map), None)
        detected[logical_name] = {
            "field": matched,
            "status": "matched" if matched else "missing",
            "candidates": candidates,
        }
    return detected


def _inspect_single_config(
    dataset_name: str,
    config_name: str,
    streaming: bool,
    samples: int,
    target_split: str | None = None,
) -> dict[str, Any]:
    builder = load_dataset_builder(dataset_name, config_name)
    builder_features = list(builder.info.features.keys()) if builder.info.features else []
    builder_splits = list(builder.info.splits.keys()) if builder.info.splits else []
    dataset = load_dataset(dataset_name, name=config_name, streaming=streaming)
    if not isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        dataset = DatasetDict({"train": dataset})

    split_names = [target_split] if target_split else list(dataset.keys())
    split_reports: list[dict[str, Any]] = []
    unified_columns: list[str] = []
    seen_columns: set[str] = set()

    for split_name in split_names:
        split_dataset = dataset[split_name]
        sample_error: str | None = None
        try:
            sample_rows = _sample_rows(split_dataset, samples)
        except Exception as exc:  # pragma: no cover - network/data dependent
            sample_rows = []
            sample_error = f"{type(exc).__name__}: {exc}"
        column_names = _column_names(split_dataset, sample_rows)
        if not column_names:
            column_names = builder_features
        for column in column_names:
            if column not in seen_columns:
                seen_columns.add(column)
                unified_columns.append(column)
        split_reports.append(
            {
                "split": split_name,
                "row_count": _row_count(split_dataset) if not streaming else None,
                "column_names": column_names,
                "samples": sample_rows,
                "sample_error": sample_error,
            }
        )

    return {
        "config": config_name,
        "splits": split_names or builder_splits,
        "builder_features": builder_features,
        "detected_fields": _detect_fields(unified_columns),
        "split_reports": split_reports,
    }


def build_report(
    dataset_name: str,
    config_name: str | None,
    streaming: bool,
    samples: int,
    target_split: str | None = None,
) -> dict[str, Any]:
    config_names = [config_name] if config_name else list(get_dataset_config_names(dataset_name))
    config_reports = [
        _inspect_single_config(dataset_name, config, streaming, samples, target_split=target_split)
        for config in config_names
    ]
    return {
        "dataset": dataset_name,
        "streaming": streaming,
        "configs": config_names,
        "target_split": target_split,
        "config_reports": config_reports,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# HF Dataset Schema Report")
    lines.append("")
    lines.append(f"- Dataset: `{report['dataset']}`")
    lines.append(f"- Streaming: `{report['streaming']}`")
    lines.append(f"- Configs: `{', '.join(report['configs'])}`")
    lines.append(f"- Target split: `{report['target_split'] or 'ALL'}`")
    for config_report in report["config_reports"]:
        lines.append("")
        lines.append(f"## Config `{config_report['config']}`")
        lines.append("")
        lines.append(f"- Splits: `{', '.join(config_report['splits'])}`")
        lines.append(f"- Builder features: `{', '.join(config_report['builder_features'])}`")
        lines.append("")
        lines.append("### Detected Fields")
        lines.append("")
        for logical_name, info in config_report["detected_fields"].items():
            lines.append(f"- `{logical_name}`: `{info['field']}` ({info['status']})")
        for split_report in config_report["split_reports"]:
            lines.append("")
            lines.append(f"### Split `{split_report['split']}`")
            lines.append("")
            lines.append(f"- Row count: `{split_report['row_count']}`")
            lines.append(f"- Column names: `{', '.join(split_report['column_names'])}`")
            if split_report.get("sample_error"):
                lines.append(f"- Sample error: `{split_report['sample_error']}`")
            lines.append("")
            lines.append("#### Sample Records")
            lines.append("")
            for index, sample in enumerate(split_report["samples"], start=1):
                lines.append(f"##### Sample {index}")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(sample, ensure_ascii=False, indent=2))
                lines.append("```")
                lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Inspect the Hugging Face legal dataset schema.")
    parser.add_argument("--dataset", default="th1nhng0/vietnamese-legal-documents")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--report-json", default=str(REPORT_JSON_PATH))
    parser.add_argument("--report-md", default=str(REPORT_MD_PATH))
    args = parser.parse_args()

    report = build_report(
        dataset_name=args.dataset,
        config_name=args.config,
        streaming=args.streaming,
        samples=max(1, args.samples),
        target_split=args.split,
    )

    report_json_path = Path(args.report_json)
    report_md_path = Path(args.report_md)
    _ensure_parent(report_json_path)
    _ensure_parent(report_md_path)
    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md_path.write_text(_render_markdown(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
