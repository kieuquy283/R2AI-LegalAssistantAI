"""
Export chunks & context chunks to Kaggle-ready embedding format.

Input:
    data/processed/chunks.jsonl
    data/processed/context_chunks.jsonl

Output:
    data/processed/legal_chunks_to_embed.jsonl

Format (JSON Lines):
    {"id": "chunk_id", "text": "embedding_text", "metadata": {...}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.ingestion.common import read_jsonl, write_jsonl


def _export_records(records: List[Dict[str, Any]], is_context: bool = False) -> List[Dict[str, Any]]:
    exported = []
    for rec in records:
        record_id = rec.get("context_chunk_id") if is_context else rec.get("chunk_id")
        text = rec.get("embedding_text") or rec.get("content") or ""
        if not text.strip():
            continue
        # Flatten all fields except the raw content (keep embedding_text as text)
        metadata = {k: v for k, v in rec.items() if k not in ("embedding_text", "content")}
        exported.append({
            "id": record_id,
            "text": text.strip(),
            "metadata": metadata,
        })
    return exported


def run_export(
    chunks_path: Path,
    context_chunks_path: Path,
    output_path: Path,
    context_output_path: Path,
) -> Dict[str, Any]:
    print(f"[INFO] Reading chunks from {chunks_path}")
    chunks = read_jsonl(chunks_path)
    print(f"[INFO] Reading context chunks from {context_chunks_path}")
    context_chunks = read_jsonl(context_chunks_path)

    exported_chunks = _export_records(chunks, is_context=False)
    exported_context = _export_records(context_chunks, is_context=True)

    print(f"[INFO] Chunks to embed: {len(exported_chunks)}")
    print(f"[INFO] Context chunks to embed: {len(exported_context)}")

    for fp, records in [(output_path, exported_chunks), (context_output_path, exported_context)]:
        fp.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(fp, records)
        print(f"[INFO] Written {len(records)} records to {fp}")

    return {
        "chunks": len(exported_chunks),
        "context_chunks": len(exported_context),
        "total": len(exported_chunks) + len(exported_context),
        "output_path": str(output_path),
        "context_output_path": str(context_output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export chunk files to Kaggle-ready embedding format.")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--context-chunks", default="data/processed/context_chunks.jsonl")
    parser.add_argument("--output", default="data/processed/legal_chunks_to_embed.jsonl")
    parser.add_argument("--context-output", default="data/processed/legal_context_chunks_to_embed.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_export(
        chunks_path=Path(args.chunks),
        context_chunks_path=Path(args.context_chunks),
        output_path=Path(args.output),
        context_output_path=Path(args.context_output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
