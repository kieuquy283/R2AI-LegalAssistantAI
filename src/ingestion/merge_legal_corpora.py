from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.ingestion.common import read_jsonl, write_json, write_jsonl


def _dedupe_rows(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(row.get(field) or "").strip() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def merge_legal_corpora(
    *,
    base_documents: str | Path = "data/processed/documents.jsonl",
    base_nodes: str | Path = "data/processed/legal_nodes.jsonl",
    base_chunks: str | Path = "data/processed/chunks.jsonl",
    base_context_chunks: str | Path = "data/processed/context_chunks.jsonl",
    base_edges: str | Path = "data/processed/legal_edges.jsonl",
    hf_documents: str | Path = "data/processed/hf_smoke_documents.jsonl",
    hf_nodes: str | Path = "data/processed/hf_smoke_legal_nodes.jsonl",
    hf_chunks: str | Path = "data/processed/hf_smoke_chunks.jsonl",
    hf_context_chunks: str | Path = "data/processed/hf_smoke_context_chunks.jsonl",
    hf_edges: str | Path = "data/processed/hf_smoke_legal_edges.jsonl",
    output_prefix: str | Path = "data/processed/merged",
) -> dict[str, Any]:
    output_prefix = Path(output_prefix)

    base_documents_rows = read_jsonl(base_documents)
    base_nodes_rows = read_jsonl(base_nodes)
    base_chunks_rows = read_jsonl(base_chunks)
    base_context_rows = read_jsonl(base_context_chunks)
    base_edges_rows = read_jsonl(base_edges)

    hf_documents_rows = read_jsonl(hf_documents)
    hf_nodes_rows = read_jsonl(hf_nodes)
    hf_chunks_rows = read_jsonl(hf_chunks)
    hf_context_rows = read_jsonl(hf_context_chunks)
    hf_edges_rows = read_jsonl(hf_edges)

    merged_documents = _dedupe_rows(base_documents_rows + hf_documents_rows, ["doc_id"])
    merged_nodes = _dedupe_rows(base_nodes_rows + hf_nodes_rows, ["node_id"])
    merged_chunks = _dedupe_rows(base_chunks_rows + hf_chunks_rows, ["chunk_id"])
    merged_context = _dedupe_rows(base_context_rows + hf_context_rows, ["context_chunk_id"])
    merged_edges = _dedupe_rows(base_edges_rows + hf_edges_rows, ["source_id", "target_id", "relation_type"])

    documents_path = output_prefix.with_name(f"{output_prefix.name}_documents.jsonl")
    nodes_path = output_prefix.with_name(f"{output_prefix.name}_legal_nodes.jsonl")
    chunks_path = output_prefix.with_name(f"{output_prefix.name}_chunks.jsonl")
    context_path = output_prefix.with_name(f"{output_prefix.name}_context_chunks.jsonl")
    edges_path = output_prefix.with_name(f"{output_prefix.name}_legal_edges.jsonl")
    report_path = output_prefix.with_name(f"{output_prefix.name}_merge_report.json")

    write_jsonl(documents_path, merged_documents)
    write_jsonl(nodes_path, merged_nodes)
    write_jsonl(chunks_path, merged_chunks)
    write_jsonl(context_path, merged_context)
    write_jsonl(edges_path, merged_edges)

    report = {
        "base_documents": len(base_documents_rows),
        "hf_documents": len(hf_documents_rows),
        "merged_documents": len(merged_documents),
        "base_nodes": len(base_nodes_rows),
        "hf_nodes": len(hf_nodes_rows),
        "merged_nodes": len(merged_nodes),
        "base_chunks": len(base_chunks_rows),
        "hf_chunks": len(hf_chunks_rows),
        "merged_chunks": len(merged_chunks),
        "base_context_chunks": len(base_context_rows),
        "hf_context_chunks": len(hf_context_rows),
        "merged_context_chunks": len(merged_context),
        "base_edges": len(base_edges_rows),
        "hf_edges": len(hf_edges_rows),
        "merged_edges": len(merged_edges),
        "documents_path": str(documents_path),
        "nodes_path": str(nodes_path),
        "chunks_path": str(chunks_path),
        "context_path": str(context_path),
        "edges_path": str(edges_path),
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge existing corpus with HF-normalized legal corpus.")
    parser.add_argument("--output-prefix", default="data/processed/merged")
    args = parser.parse_args()
    report = merge_legal_corpora(output_prefix=args.output_prefix)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
