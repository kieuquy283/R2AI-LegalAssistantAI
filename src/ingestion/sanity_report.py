from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from src.ingestion.common import write_json, read_jsonl


DEFAULT_DOCUMENTS_PATH = Path("data/processed/documents.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_CONTEXT_CHUNKS_PATH = Path("data/processed/context_chunks.jsonl")
DEFAULT_EDGES_PATH = Path("data/processed/legal_edges.jsonl")
DEFAULT_EXPLICIT_REFS_PATH = Path("data/processed/explicit_refs.jsonl")
DEFAULT_CROSS_DOMAIN_EDGES_PATH = Path("data/processed/cross_domain_edges.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/sanity_report.json")


def build_sanity_report(
    documents: List[Dict[str, object]],
    chunks: List[Dict[str, object]],
    context_chunks: List[Dict[str, object]],
    edges: List[Dict[str, object]],
    explicit_refs: List[Dict[str, object]],
    cross_domain_edges: List[Dict[str, object]],
) -> Dict[str, object]:
    def normalize_for_noise(text: str) -> str:
        lowered = (text or "").lower()
        return lowered.replace("đ", "d")

    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    node_parent_issues = [chunk["chunk_id"] for chunk in chunks if chunk.get("parent_id") is None]
    missing_context = [chunk["chunk_id"] for chunk in chunks if not chunk.get("context_chunk_id")]
    dangling_prev_next = [
        chunk["chunk_id"]
        for chunk in chunks
        if (chunk.get("prev_chunk_id") and chunk["prev_chunk_id"] not in chunk_ids)
        or (chunk.get("next_chunk_id") and chunk["next_chunk_id"] not in chunk_ids)
    ]
    noise_hits = [
        chunk["chunk_id"]
        for chunk in chunks
        if "dang theo doi" in normalize_for_noise(str(chunk.get("content") or ""))
    ]
    unresolved_refs = [ref for ref in explicit_refs if ref.get("resolution") != "resolved"]
    domains: Dict[str, int] = {}
    for document in documents:
        domain = str(document.get("domain") or "unknown")
        domains[domain] = domains.get(domain, 0) + 1

    return {
        "ok": not missing_context and not dangling_prev_next and not noise_hits,
        "summary": {
            "documents": len(documents),
            "chunks": len(chunks),
            "context_chunks": len(context_chunks),
            "legal_edges": len(edges),
            "explicit_refs": len(explicit_refs),
            "cross_domain_edges": len(cross_domain_edges),
        },
        "domains": domains,
        "critical_issues": {
            "missing_context_chunk_ids": missing_context,
            "dangling_prev_next": dangling_prev_next,
            "noise_hits": noise_hits,
        },
        "warnings": {
            "chunks_without_parent": node_parent_issues[:25],
            "unresolved_explicit_refs": unresolved_refs[:25],
        },
    }


def run_sanity_report(
    *,
    documents_path: str | Path = DEFAULT_DOCUMENTS_PATH,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    context_chunks_path: str | Path = DEFAULT_CONTEXT_CHUNKS_PATH,
    edges_path: str | Path = DEFAULT_EDGES_PATH,
    explicit_refs_path: str | Path = DEFAULT_EXPLICIT_REFS_PATH,
    cross_domain_edges_path: str | Path = DEFAULT_CROSS_DOMAIN_EDGES_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Dict[str, object]:
    report = build_sanity_report(
        documents=read_jsonl(documents_path),
        chunks=read_jsonl(chunks_path),
        context_chunks=read_jsonl(context_chunks_path),
        edges=read_jsonl(edges_path),
        explicit_refs=read_jsonl(explicit_refs_path),
        cross_domain_edges=read_jsonl(cross_domain_edges_path),
    )
    write_json(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ingestion sanity report.")
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS_PATH))
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--context-chunks", default=str(DEFAULT_CONTEXT_CHUNKS_PATH))
    parser.add_argument("--edges", default=str(DEFAULT_EDGES_PATH))
    parser.add_argument("--explicit-refs", default=str(DEFAULT_EXPLICIT_REFS_PATH))
    parser.add_argument("--cross-domain-edges", default=str(DEFAULT_CROSS_DOMAIN_EDGES_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_sanity_report(
        documents_path=args.documents,
        chunks_path=args.chunks,
        context_chunks_path=args.context_chunks,
        edges_path=args.edges,
        explicit_refs_path=args.explicit_refs,
        cross_domain_edges_path=args.cross_domain_edges,
        output_path=args.output,
    )
    print(f"Sanity report: DONE (ok={report['ok']}, chunks={report['summary']['chunks']})")


if __name__ == "__main__":
    main()
