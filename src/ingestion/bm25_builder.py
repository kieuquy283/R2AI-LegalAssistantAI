from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from rag.modules.retrieval.utils import tokenize_for_bm25
from src.ingestion.common import ensure_parent, read_jsonl


DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_CORPUS_PATH = Path("data/indexes/bm25_corpus.json")
DEFAULT_METADATA_PATH = Path("data/indexes/bm25_metadata.json")


def build_bm25_artifacts(chunks: List[Dict[str, object]]) -> tuple[list[dict], list[dict]]:
    corpus: List[Dict[str, object]] = []
    metadata: List[Dict[str, object]] = []
    for index, chunk in enumerate(chunks):
        tokens = tokenize_for_bm25(str(chunk.get("embedding_text") or chunk.get("content") or ""))
        corpus.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk.get("embedding_text") or chunk.get("content"),
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
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> int:
    chunks = read_jsonl(chunks_path)
    corpus, metadata = build_bm25_artifacts(chunks)
    ensure_parent(corpus_path).write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    ensure_parent(metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(corpus)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BM25 corpus artifacts from chunks.")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = run_bm25_builder(chunks_path=args.chunks, corpus_path=args.corpus, metadata_path=args.metadata)
    print(f"BM25 build: DONE ({count} rows)")


if __name__ == "__main__":
    main()
