from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
import uuid

from rag.utils.hashes import sha256_text


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_chunk_metadata(
    chunk_id: str,
    source_file: str,
    file_hash: str,
    chunk_text: str,
    chunk_index: int,
) -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "file_hash": file_hash,
        "content_hash": sha256_text(chunk_text),
        "chunk_index": chunk_index,
        "is_active": True,
        "created_at": utc_now_iso(),
        "text": chunk_text,
    }


def build_chunk_metadata_list(
    source_file: str,
    file_hash: str,
    chunks: List[str],
) -> List[Dict[str, Any]]:
    metadata_list: List[Dict[str, Any]] = []
    for i, chunk_text in enumerate(chunks):
        metadata_list.append(
            make_chunk_metadata(
                chunk_id=str(uuid.uuid4()),
                source_file=source_file,
                file_hash=file_hash,
                chunk_text=chunk_text,
                chunk_index=i,
            )
        )
    return metadata_list


def deactivate_chunks_for_file(
    metadata: List[Dict[str, Any]],
    source_file: str,
) -> int:
    changed = 0
    for item in metadata:
        if item.get("source_file") == source_file and item.get("is_active", True):
            item["is_active"] = False
            item["deactivated_at"] = utc_now_iso()
            changed += 1
    return changed
