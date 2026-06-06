from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from rag.retrieval.vectorstore import get_embeddings
from src.ingestion.common import ensure_dir, ensure_parent, read_jsonl


DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_FAISS_PATH = Path("data/indexes/faiss.index")
DEFAULT_METADATA_PATH = Path("data/indexes/chunk_metadata.json")


def _build_metadata_row(index: int, chunk: Dict[str, object]) -> Dict[str, object]:
    return {
        "index": index,
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "domain": chunk["domain"],
        "doc_title": chunk.get("doc_title"),
        "article": chunk.get("article"),
        "clause": chunk.get("clause"),
        "citation": chunk.get("citation"),
        "source_url": chunk.get("source_url"),
        "context_chunk_id": chunk.get("context_chunk_id"),
        "parent_id": chunk.get("parent_id"),
        "prev_chunk_id": chunk.get("prev_chunk_id"),
        "next_chunk_id": chunk.get("next_chunk_id"),
    }


def build_index(chunks: List[Dict[str, object]]) -> Tuple[faiss.Index, List[Dict[str, object]]]:
    embedding_model = get_embeddings()
    texts = [str(chunk["embedding_text"]) for chunk in chunks]
    vectors = np.array(embedding_model.embed_documents(texts), dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Unexpected embedding output shape")

    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    metadata = [_build_metadata_row(i, chunk) for i, chunk in enumerate(chunks)]
    return index, metadata


def run_index_builder(
    *,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    faiss_index_path: str | Path = DEFAULT_FAISS_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> int:
    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError("No chunks available for index building")

    ensure_dir(Path(faiss_index_path).parent)
    index, metadata = build_index(chunks)
    faiss.write_index(index, str(ensure_parent(faiss_index_path)))
    ensure_parent(metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from chunks.jsonl.")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--faiss-index", default=str(DEFAULT_FAISS_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = run_index_builder(
        chunks_path=args.chunks,
        faiss_index_path=args.faiss_index,
        metadata_path=args.metadata,
    )
    print(f"FAISS index build: DONE ({count} vectors)")


if __name__ == "__main__":
    main()
