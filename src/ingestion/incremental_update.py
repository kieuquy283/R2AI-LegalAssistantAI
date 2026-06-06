from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.ingestion.bm25_builder import run_bm25_builder
from src.ingestion.common import read_jsonl, write_json, write_jsonl
from src.ingestion.document_parser import build_document_record
from src.ingestion.index_builder import run_index_builder
from src.ingestion.legal_chunker import build_chunks
from src.ingestion.legal_structure_parser import parse_document_structure
from src.ingestion.reference_enricher import run_reference_enricher
from src.ingestion.sanity_report import run_sanity_report
from src.ingestion.text_cleaner import build_cleaned_record


DEFAULT_STATE_PATH = Path("data/processed/incremental_state.json")
DEFAULT_MANIFEST_PATH = Path("data/raw/documents_manifest.jsonl")


def _load_state(path: str | Path) -> Dict[str, object]:
    state_path = Path(path)
    if not state_path.exists():
        return {"manifest": {}}
    return json.loads(state_path.read_text(encoding="utf-8"))


def diff_manifest(
    manifest_rows: List[Dict[str, object]],
    previous_state: Dict[str, object],
) -> Tuple[List[str], List[str], List[str], Dict[str, Dict[str, object]]]:
    current_map = {
        str(row["doc_id"]): {
            "content_hash": row.get("content_hash"),
            "html_hash": row.get("html_hash"),
            "success": row.get("success"),
        }
        for row in manifest_rows
        if row.get("success")
    }
    previous_map = dict(previous_state.get("manifest") or {})

    added = [doc_id for doc_id in current_map if doc_id not in previous_map]
    changed = [
        doc_id
        for doc_id, info in current_map.items()
        if doc_id in previous_map and info != previous_map.get(doc_id)
    ]
    removed = [doc_id for doc_id in previous_map if doc_id not in current_map]
    return sorted(added), sorted(changed), sorted(removed), current_map


def _merge_records(existing_rows: List[Dict[str, object]], updated_rows: List[Dict[str, object]], removed_doc_ids: List[str]) -> List[Dict[str, object]]:
    updated_map = {str(row["doc_id"]): row for row in updated_rows}
    merged: List[Dict[str, object]] = []
    for row in existing_rows:
        doc_id = str(row["doc_id"])
        if doc_id in removed_doc_ids or doc_id in updated_map:
            continue
        merged.append(row)
    merged.extend(updated_rows)
    merged.sort(key=lambda row: str(row["doc_id"]))
    return merged


def run_incremental_update(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    state_path: str | Path = DEFAULT_STATE_PATH,
) -> Dict[str, object]:
    manifest_rows = read_jsonl(manifest_path)
    previous_state = _load_state(state_path)
    added, changed, removed, current_map = diff_manifest(manifest_rows, previous_state)
    changed_doc_ids = sorted(set(added + changed))

    status = "no_changes"
    if changed_doc_ids or removed:
        status = "updated"
        success_manifest_rows = [row for row in manifest_rows if row.get("success") and row.get("doc_id") in changed_doc_ids]

        updated_documents = [build_document_record(row) for row in success_manifest_rows]
        existing_documents = read_jsonl("data/processed/documents.jsonl") if Path("data/processed/documents.jsonl").exists() else []
        merged_documents = _merge_records(existing_documents, updated_documents, removed)
        write_jsonl("data/processed/documents.jsonl", merged_documents)

        updated_cleaned = [build_cleaned_record(document) for document in updated_documents]
        existing_cleaned = read_jsonl("data/processed/cleaned_documents.jsonl") if Path("data/processed/cleaned_documents.jsonl").exists() else []
        merged_cleaned = _merge_records(existing_cleaned, updated_cleaned, removed)
        write_jsonl("data/processed/cleaned_documents.jsonl", merged_cleaned)

        updated_nodes: List[Dict[str, object]] = []
        for cleaned in updated_cleaned:
            updated_nodes.extend(parse_document_structure(cleaned))
        existing_nodes = read_jsonl("data/processed/legal_nodes.jsonl") if Path("data/processed/legal_nodes.jsonl").exists() else []
        merged_nodes = [
            row
            for row in existing_nodes
            if str(row["doc_id"]) not in set(changed_doc_ids + removed)
        ]
        merged_nodes.extend(updated_nodes)
        merged_nodes.sort(key=lambda row: (str(row["doc_id"]), int(row.get("start_char", 0))))
        write_jsonl("data/processed/legal_nodes.jsonl", merged_nodes)

        chunks, context_chunks, edges = build_chunks(merged_nodes, merged_documents)
        write_jsonl("data/processed/chunks.jsonl", chunks)
        write_jsonl("data/processed/context_chunks.jsonl", context_chunks)
        write_jsonl("data/processed/legal_edges.jsonl", edges)

        run_reference_enricher()
        run_bm25_builder()
        run_index_builder()
        run_sanity_report()

    new_state = {
        "manifest": current_map,
        "last_status": status,
        "changed_doc_ids": changed_doc_ids,
        "removed_doc_ids": removed,
    }
    write_json(state_path, new_state)
    return new_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally refresh processed ingestion artifacts from manifest deltas.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = run_incremental_update(manifest_path=args.manifest, state_path=args.state)
    print(
        "Incremental update: DONE "
        f"(status={state['last_status']}, changed={len(state['changed_doc_ids'])}, removed={len(state['removed_doc_ids'])})"
    )


if __name__ == "__main__":
    main()
