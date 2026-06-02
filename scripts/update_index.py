from __future__ import annotations

import argparse

from rag.config.retrieval import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_INDEX_DIR
from rag.pipelines.indexing_pipeline import run_update_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally update FAISS index from documents.")
    parser.add_argument("--data-dir", default="data", help="Directory containing source documents.")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Directory to store FAISS index.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    run_update_documents(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
