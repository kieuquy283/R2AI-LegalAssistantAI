from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_community.vectorstores import FAISS

from rag.config.retrieval import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_CORPUS_JSON,
    DEFAULT_INDEX_DIR,
)
from rag.ingestion.chunkers import chunk_document
from rag.ingestion.loaders import scan_document_files
from rag.ingestion.metadata import (
    build_chunk_metadata_list,
    deactivate_chunks_for_file,
    utc_now_iso,
)
from rag.retrieval.vectorstore import get_embeddings
from rag.utils.hashes import sha256_file, sha256_text
from rag.utils.io import load_json, save_json
from rag.utils.logging import get_logger


logger = get_logger(__name__)

MANIFEST_FILENAME = "manifest.json"
CHUNKS_METADATA_FILENAME = "chunks_metadata.json"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def index_exists(index_dir: str | Path) -> bool:
    index_dir = Path(index_dir)
    return (
        (index_dir / "index.faiss").exists()
        and (index_dir / "index.pkl").exists()
        and (index_dir / MANIFEST_FILENAME).exists()
        and (index_dir / CHUNKS_METADATA_FILENAME).exists()
    )


def build_faiss_in_batches(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    embedding_model: Any,
    batch_size: int = 100,
) -> FAISS:
    if not texts:
        raise ValueError("No texts to build FAISS.")
    if len(texts) != len(metadatas):
        raise ValueError("texts và metadatas phải cùng độ dài.")

    vectorstore = FAISS.from_texts(
        texts=texts[:batch_size],
        embedding=embedding_model,
        metadatas=metadatas[:batch_size],
    )

    for start in range(batch_size, len(texts), batch_size):
        end = start + batch_size
        logger.info("Embedding batch %s -> %s/%s", start, min(end, len(texts)), len(texts))
        vectorstore.add_texts(
            texts=texts[start:end],
            metadatas=metadatas[start:end],
        )

    return vectorstore


def build_chunks_and_metadata_for_file(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[str], List[Dict[str, Any]], str]:
    file_hash = sha256_file(file_path)
    chunks = chunk_document(
        file_path=file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    metadata = build_chunk_metadata_list(
        source_file=file_path,
        file_hash=file_hash,
        chunks=chunks,
    )
    return chunks, metadata, file_hash


def build_index_from_documents(
    data_dir: str,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    embedding_model: Optional[Any] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    embedding_model = embedding_model or get_embeddings()
    ensure_dir(index_dir)

    manifest_path = Path(index_dir) / MANIFEST_FILENAME
    metadata_path = Path(index_dir) / CHUNKS_METADATA_FILENAME

    files = scan_document_files(data_dir)
    if not files:
        raise ValueError(f"No supported documents found in: {data_dir}")

    all_texts: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {}

    for file_path in files:
        texts, metadatas, file_hash = build_chunks_and_metadata_for_file(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_texts.extend(texts)
        all_metadatas.extend(metadatas)
        manifest[file_path] = {
            "file_hash": file_hash,
            "last_indexed_at": utc_now_iso(),
        }
        logger.info("[BUILD] %s -> %s chunks", file_path, len(texts))

    if not all_texts:
        raise ValueError("No chunks generated from the provided documents.")

    vectorstore = build_faiss_in_batches(
        texts=all_texts,
        metadatas=all_metadatas,
        embedding_model=embedding_model,
        batch_size=100,
    )
    vectorstore.save_local(str(index_dir))

    save_json(manifest_path, manifest)
    save_json(metadata_path, all_metadatas)

    logger.info(
        "[BUILD DONE] files=%s, chunks=%s, index_dir=%s",
        len(files),
        len(all_texts),
        index_dir,
    )


def update_index_from_documents(
    data_dir: str,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    embedding_model: Optional[Any] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    embedding_model = embedding_model or get_embeddings()

    if not index_exists(index_dir):
        raise ValueError(
            f"Index not found in '{index_dir}'. Run build_index_from_documents first."
        )

    manifest_path = Path(index_dir) / MANIFEST_FILENAME
    metadata_path = Path(index_dir) / CHUNKS_METADATA_FILENAME

    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    manifest: Dict[str, Any] = load_json(manifest_path, {})
    metadata: List[Dict[str, Any]] = load_json(metadata_path, [])

    current_files = scan_document_files(data_dir)
    current_file_set = set(current_files)
    old_file_set = set(manifest.keys())

    new_texts: List[str] = []
    new_metadatas: List[Dict[str, Any]] = []

    added_files = 0
    updated_files = 0
    deleted_files = 0
    deactivated_chunks = 0

    for file_path in current_files:
        current_hash = sha256_file(file_path)

        if file_path not in manifest:
            texts, metadatas, file_hash = build_chunks_and_metadata_for_file(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            new_texts.extend(texts)
            new_metadatas.extend(metadatas)
            manifest[file_path] = {
                "file_hash": file_hash,
                "last_indexed_at": utc_now_iso(),
            }
            added_files += 1
            logger.info("[UPDATE][NEW] %s -> %s chunks", file_path, len(texts))
            continue

        if manifest[file_path].get("file_hash") != current_hash:
            deactivated_chunks += deactivate_chunks_for_file(metadata, file_path)

            texts, metadatas, file_hash = build_chunks_and_metadata_for_file(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            new_texts.extend(texts)
            new_metadatas.extend(metadatas)
            manifest[file_path] = {
                "file_hash": file_hash,
                "last_indexed_at": utc_now_iso(),
            }
            updated_files += 1
            logger.info("[UPDATE][MODIFIED] %s -> %s new chunks", file_path, len(texts))

    removed_files = old_file_set - current_file_set
    for file_path in sorted(removed_files):
        deactivated_chunks += deactivate_chunks_for_file(metadata, file_path)
        manifest.pop(file_path, None)
        deleted_files += 1
        logger.info("[UPDATE][DELETED] %s", file_path)

    if new_texts:
        vectorstore.add_texts(texts=new_texts, metadatas=new_metadatas)
        metadata.extend(new_metadatas)

    vectorstore.save_local(str(index_dir))
    save_json(manifest_path, manifest)
    save_json(metadata_path, metadata)

    logger.info(
        "[UPDATE DONE] added_files=%s, updated_files=%s, deleted_files=%s, "
        "deactivated_chunks=%s, new_chunks=%s",
        added_files,
        updated_files,
        deleted_files,
        deactivated_chunks,
        len(new_texts),
    )


def build_index_from_corpus_json(
    corpus_json_path: str | Path = DEFAULT_CORPUS_JSON,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    embedding_model: Optional[Any] = None,
) -> None:
    embedding_model = embedding_model or get_embeddings()
    ensure_dir(index_dir)

    corpus_path = Path(corpus_json_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus JSON not found: {corpus_json_path}")

    manifest_path = Path(index_dir) / MANIFEST_FILENAME
    metadata_path = Path(index_dir) / CHUNKS_METADATA_FILENAME

    data = load_json(corpus_path, [])
    if not isinstance(data, list) or not data:
        raise ValueError(f"Corpus JSON is empty or invalid: {corpus_json_path}")

    all_texts: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    corpus_file_hash = sha256_file(corpus_path)

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        if not text:
            continue

        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": str(metadata)}

        enriched_metadata = {
            "chunk_id": str(i),
            "source_file": str(corpus_path),
            "file_hash": corpus_file_hash,
            "content_hash": sha256_text(text),
            "chunk_index": i,
            "is_active": True,
            "created_at": utc_now_iso(),
            "text": text,
            **metadata,
        }

        all_texts.append(text)
        all_metadatas.append(enriched_metadata)

    if not all_texts:
        raise ValueError("No valid texts found in corpus JSON.")

    vectorstore = build_faiss_in_batches(
        texts=all_texts,
        metadatas=all_metadatas,
        embedding_model=embedding_model,
        batch_size=100,
    )
    vectorstore.save_local(str(index_dir))

    manifest = {
        str(corpus_path): {
            "file_hash": corpus_file_hash,
            "last_indexed_at": utc_now_iso(),
            "source_type": "retrieval_corpus_json",
            "num_chunks": len(all_texts),
        }
    }

    save_json(manifest_path, manifest)
    save_json(metadata_path, all_metadatas)

    logger.info(
        "[BUILD JSON DONE] corpus=%s, chunks=%s, index_dir=%s",
        corpus_json_path,
        len(all_texts),
        index_dir,
    )


def load_chunks_metadata(index_dir: str | Path = DEFAULT_INDEX_DIR) -> List[Dict[str, Any]]:
    metadata_path = Path(index_dir) / CHUNKS_METADATA_FILENAME
    return load_json(metadata_path, []) or []
