from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from rag.retrieval.vectorstore import get_embeddings
from src.ingestion.common import ensure_dir, ensure_parent, read_jsonl


DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_CONTEXT_CHUNKS_PATH = Path("data/processed/context_chunks.jsonl")
DEFAULT_CLEANED_DOCUMENTS_PATH = Path("data/processed/cleaned_documents.jsonl")
DEFAULT_FAISS_PATH = Path("data/indexes/faiss.index")
DEFAULT_METADATA_PATH = Path("data/indexes/chunk_metadata.json")
DEFAULT_BATCH_SIZE = 32
DEFAULT_SOURCE_MAX_CHARS = 2000


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


def _embedding_text_for_row(row: Dict[str, object]) -> str:
    if row.get("citation") and row.get("content"):
        return f"{row['citation']}\n{row['content']}"
    return str(row.get("embedding_text") or row.get("content") or row.get("citation") or "")


def _encode_texts(
    embedding_model: object,
    texts: List[str],
    *,
    batch_size: int,
) -> np.ndarray:
    if hasattr(embedding_model, "model") and getattr(embedding_model, "model", None) is not None:
        vectors = embedding_model.model.encode(
            [f"passage: {text}" for text in texts],
            batch_size=min(batch_size, 64),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    else:
        vectors = embedding_model.embed_documents(texts)
    return np.asarray(vectors, dtype="float32")


def build_index(
    chunks: List[Dict[str, object]],
    *,
    cleaned_documents: List[Dict[str, object]] | None = None,
    context_chunks: List[Dict[str, object]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[faiss.Index, List[Dict[str, object]]]:
    embedding_model = get_embeddings()
    metadata: List[Dict[str, object]] = []
    index: faiss.Index | None = None

    source_rows: List[Dict[str, object]] = []
    source_keys: List[str] = []
    source_texts: List[str] = []
    source_max_chars = int(os.getenv("EMBEDDING_SOURCE_MAX_CHARS", str(DEFAULT_SOURCE_MAX_CHARS)))

    if cleaned_documents:
        for row in cleaned_documents:
            key = str(row.get("doc_id") or "")
            if not key:
                continue
            source_keys.append(key)
            source_rows.append(row)
            source_texts.append(str(row.get("cleaned_text") or "")[:source_max_chars])

    if not source_rows and context_chunks:
        for row in context_chunks:
            key = str(row.get("context_chunk_id") or row.get("chunk_id") or "")
            if not key:
                continue
            source_keys.append(key)
            source_rows.append(row)
            source_texts.append(_embedding_text_for_row(row))

    if not source_rows:
        for chunk in chunks:
            key = str(chunk.get("doc_id") or chunk.get("chunk_id") or "")
            if not key:
                continue
            source_keys.append(key)
            source_rows.append(chunk)
            source_texts.append(_embedding_text_for_row(chunk))

    vector_map: Dict[str, np.ndarray] = {}
    for batch_start in range(0, len(source_rows), batch_size):
        batch_rows = source_rows[batch_start: batch_start + batch_size]
        batch_texts = source_texts[batch_start: batch_start + batch_size]
        vectors = _encode_texts(embedding_model, batch_texts, batch_size=batch_size)

        if vectors.ndim != 2 or vectors.shape[0] != len(batch_rows):
            raise ValueError("Unexpected embedding output shape")

        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])

        for row, vector in zip(batch_rows, vectors, strict=True):
            row_key = str(row.get("doc_id") or row.get("context_chunk_id") or row.get("chunk_id") or "")
            vector_map[row_key] = vector

    if index is None:
        raise ValueError("No vectors were produced while building the index")

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        vectors = np.stack(
            [
                vector_map[
                    str(chunk.get("doc_id") or chunk.get("context_chunk_id") or chunk.get("chunk_id") or "")
                ]
                for chunk in batch
            ],
            axis=0,
        ).astype("float32", copy=False)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        metadata.extend(_build_metadata_row(batch_start + i, chunk) for i, chunk in enumerate(batch))

    return index, metadata


def run_index_builder(
    *,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    cleaned_documents_path: str | Path = DEFAULT_CLEANED_DOCUMENTS_PATH,
    context_chunks_path: str | Path = DEFAULT_CONTEXT_CHUNKS_PATH,
    faiss_index_path: str | Path = DEFAULT_FAISS_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    chunks = read_jsonl(chunks_path)
    cleaned_documents = read_jsonl(cleaned_documents_path) if Path(cleaned_documents_path).exists() else []
    context_chunks = read_jsonl(context_chunks_path) if Path(context_chunks_path).exists() else []
    if not chunks:
        raise ValueError("No chunks available for index building")

    ensure_dir(Path(faiss_index_path).parent)
    index, metadata = build_index(
        chunks,
        cleaned_documents=cleaned_documents,
        context_chunks=context_chunks,
        batch_size=batch_size,
    )
    faiss.write_index(index, str(ensure_parent(faiss_index_path)))
    ensure_parent(metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from chunks.jsonl.")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--cleaned-documents", default=str(DEFAULT_CLEANED_DOCUMENTS_PATH))
    parser.add_argument("--context-chunks", default=str(DEFAULT_CONTEXT_CHUNKS_PATH))
    parser.add_argument("--faiss-index", default=str(DEFAULT_FAISS_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = run_index_builder(
        chunks_path=args.chunks,
        cleaned_documents_path=args.cleaned_documents,
        context_chunks_path=args.context_chunks,
        faiss_index_path=args.faiss_index,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
    )
    print(f"FAISS index build: DONE ({count} vectors)")


if __name__ == "__main__":
    main()
