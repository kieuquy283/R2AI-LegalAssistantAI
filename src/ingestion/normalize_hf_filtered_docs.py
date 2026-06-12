from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.ingestion.common import normalize_text, read_jsonl, write_json, write_jsonl
from src.ingestion.legal_chunker import build_chunks
from src.ingestion.legal_structure_parser import parse_document_structure


def _to_document_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned_text = normalize_text(str(row.get("content") or ""))
    return {
        "doc_id": str(row.get("doc_id") or "").strip(),
        "doc_slug": str(row.get("doc_slug") or "").strip(),
        "doc_title": str(row.get("doc_title") or "").strip(),
        "doc_number": str(row.get("doc_number") or "").strip(),
        "doc_type": str(row.get("doc_type") or "").strip(),
        "issuer": str(row.get("issuer") or "").strip(),
        "issue_date": str(row.get("issued_date") or "").strip(),
        "effective_date": str(row.get("effective_date") or "").strip(),
        "domain": str(row.get("domain") or "").strip(),
        "candidate_domains": list(row.get("candidate_domains") or []),
        "source_url": str(row.get("source_url") or "").strip(),
        "source_dataset": str(row.get("source_dataset") or "").strip(),
        "source_id": str(row.get("source_id") or "").strip(),
        "matched_group": str(row.get("matched_group") or "").strip(),
        "matched_keywords": list(row.get("matched_keywords") or []),
        "priority": int(row.get("priority") or 0),
        "content_hash": str(row.get("content_hash") or "").strip(),
        "cleaned_text": cleaned_text,
    }


def normalize_hf_filtered_docs(input_path: str | Path, output_prefix: str | Path) -> dict[str, Any]:
    filtered_rows = read_jsonl(input_path)
    output_prefix = Path(output_prefix)

    documents = [_to_document_row(row) for row in filtered_rows]
    legal_nodes: list[dict[str, Any]] = []
    for document in documents:
        legal_nodes.extend(parse_document_structure(document))
    chunks, context_chunks, legal_edges = build_chunks(legal_nodes, documents)

    documents_path = output_prefix.with_name(f"{output_prefix.name}_documents.jsonl")
    nodes_path = output_prefix.with_name(f"{output_prefix.name}_legal_nodes.jsonl")
    chunks_path = output_prefix.with_name(f"{output_prefix.name}_chunks.jsonl")
    context_path = output_prefix.with_name(f"{output_prefix.name}_context_chunks.jsonl")
    edges_path = output_prefix.with_name(f"{output_prefix.name}_legal_edges.jsonl")
    report_path = output_prefix.with_name(f"{output_prefix.name}_normalize_report.json")

    write_jsonl(documents_path, documents)
    write_jsonl(nodes_path, legal_nodes)
    write_jsonl(chunks_path, chunks)
    write_jsonl(context_path, context_chunks)
    write_jsonl(edges_path, legal_edges)

    report = {
        "input_path": str(input_path),
        "documents_path": str(documents_path),
        "nodes_path": str(nodes_path),
        "chunks_path": str(chunks_path),
        "context_chunks_path": str(context_path),
        "edges_path": str(edges_path),
        "documents": len(documents),
        "legal_nodes": len(legal_nodes),
        "chunks": len(chunks),
        "context_chunks": len(context_chunks),
        "legal_edges": len(legal_edges),
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize HF-filtered legal docs into the current RAG schema.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    report = normalize_hf_filtered_docs(args.input, args.output_prefix)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
