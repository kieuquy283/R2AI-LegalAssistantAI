from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from rag.modules.retrieval.utils import tokenize_for_bm25
from src.ingestion.common import ensure_parent, read_jsonl


DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_DOCUMENTS_PATH = Path("data/processed/cleaned_documents.jsonl")
DEFAULT_CORPUS_PATH = Path("data/indexes/bm25_corpus.json")
DEFAULT_METADATA_PATH = Path("data/indexes/bm25_metadata.json")


def _combined_text(row: Dict[str, object]) -> str:
    parts = [
        row.get("doc_title"),
        row.get("domain"),
        row.get("legal_path"),
        row.get("citation"),
        row.get("article"),
        row.get("clause"),
        row.get("content"),
        row.get("cleaned_text"),
        row.get("embedding_text"),
    ]
    values = []
    for part in parts:
        text = str(part or "").strip()
        if text:
            values.append(text)
    return "\n".join(values).strip()


def build_bm25_artifacts(
    chunks: List[Dict[str, object]],
    documents: List[Dict[str, object]] | None = None,
) -> tuple[list[dict], list[dict]]:
    first_chunk_by_doc: Dict[str, str] = {}
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id") or "")
        if doc_id and doc_id not in first_chunk_by_doc:
            first_chunk_by_doc[doc_id] = str(chunk.get("chunk_id") or "")

    if documents:
        corpus: List[Dict[str, object]] = []
        metadata: List[Dict[str, object]] = []
        for index, document in enumerate(documents):
            doc_id = str(document.get("doc_id") or "")
            text = _combined_text(document)[:4000]
            chunk_id = first_chunk_by_doc.get(doc_id) or doc_id
            tokens = tokenize_for_bm25(text)
            corpus.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": text,
                    "tokens": tokens,
                }
            )
            metadata.append(
                {
                    "index": index,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "domain": document.get("domain"),
                    "citation": document.get("doc_title") or doc_id,
                    "context_chunk_id": None,
                }
            )
        return corpus, metadata

    corpus: List[Dict[str, object]] = []
    metadata: List[Dict[str, object]] = []
    for index, chunk in enumerate(chunks):
        text = _combined_text(chunk)
        tokens = tokenize_for_bm25(text)
        corpus.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": text,
                "tokens": tokens,
            }
        )
        metadata.append(
            {
                "index": index,
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "domain": chunk["domain"],
                "citation": chunk.get("citation"),
                "context_chunk_id": chunk.get("context_chunk_id"),
            }
        )
    return corpus, metadata


def run_bm25_builder(
    *,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    documents_path: str | Path = DEFAULT_DOCUMENTS_PATH,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> int:
    chunks = read_jsonl(chunks_path)
    documents = read_jsonl(documents_path) if Path(documents_path).exists() else []
    corpus, metadata = build_bm25_artifacts(chunks, documents=documents)
    ensure_parent(corpus_path).write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    ensure_parent(metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(corpus)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BM25 corpus artifacts from chunks.")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS_PATH))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = run_bm25_builder(
        chunks_path=args.chunks,
        documents_path=args.documents,
        corpus_path=args.corpus,
        metadata_path=args.metadata,
    )
    print(f"BM25 build: DONE ({count} rows)")


if __name__ == "__main__":
    main()
