from __future__ import annotations

from pathlib import Path

from rag.ingestion.indexing import (
    build_index_from_corpus_json,
    build_index_from_documents,
    update_index_from_documents,
)


def run_build_documents(
    data_dir: str,
    index_dir: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    build_index_from_documents(
        data_dir=data_dir,
        index_dir=index_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def run_update_documents(
    data_dir: str,
    index_dir: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    update_index_from_documents(
        data_dir=data_dir,
        index_dir=index_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def run_build_from_json(
    corpus_json: str,
    index_dir: str,
) -> None:
    build_index_from_corpus_json(
        corpus_json_path=corpus_json,
        index_dir=index_dir,
    )
