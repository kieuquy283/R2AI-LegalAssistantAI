from __future__ import annotations

import argparse

from rag.config.retrieval import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_CORPUS_JSON, DEFAULT_INDEX_DIR
from rag.pipelines.indexing_pipeline import run_build_documents, run_build_from_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index.")
    parser.add_argument(
        "--mode",
        choices=["documents", "from_json"],
        default="documents",
        help="Build from source documents or directly from retrieval corpus JSON.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing source documents.")
    parser.add_argument("--corpus-json", default=str(DEFAULT_CORPUS_JSON), help="Path to retrieval corpus JSON.")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Directory to store FAISS index.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)

    args = parser.parse_args()

    if args.mode == "documents":
        run_build_documents(
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    else:
        run_build_from_json(
            corpus_json=args.corpus_json,
            index_dir=args.index_dir,
        )


if __name__ == "__main__":
    main()
