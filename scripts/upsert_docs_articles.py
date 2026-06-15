from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Generator
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, PayloadSchemaType
from sentence_transformers import SentenceTransformer

import random

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


def _ensure_cache() -> None:
    os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"D:\huggingface_cache\hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\huggingface_cache\transformers")
    os.environ.setdefault("TORCH_HOME", r"D:\huggingface_cache\torch")


def _to_qdrant_id(raw_id: object) -> int | UUID:
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str):
        try:
            return int(raw_id)
        except ValueError:
            pass
        try:
            return UUID(raw_id)
        except ValueError:
            pass
    digest = hashlib.sha256(str(raw_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & 0x7FFFFFFFFFFFFFFF


def _stream_jsonl(file_path: Path) -> Generator[dict, None, None]:
    if str(file_path).endswith(".gz"):
        open_fn = gzip.open
    else:
        open_fn = open
    with open_fn(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _embed_batch(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    texts = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=len(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def _record_to_point(record: dict[str, Any], vector: list[float]) -> PointStruct | None:
    raw_id = record.get("id") or record.get("doc_id") or record.get("node_id")
    if raw_id is None:
        return None
    point_id = _to_qdrant_id(raw_id)
    payload: dict[str, Any] = {}
    for k, v in record.items():
        if k in ("vector", "embedding", "id", "doc_id", "node_id"):
            continue
        if k == "metadata" and isinstance(v, dict):
            payload.update(v)
        else:
            payload[k] = v
    return PointStruct(id=point_id, vector=vector, payload=payload)


def _upsert_streaming(
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer | None,
    file_path: Path,
    embed_batch_size: int,
    upsert_batch_size: int,
) -> int:
    total_upserted = 0
    total_processed = 0
    skipped_no_vector = 0
    
    points_buffer: list[PointStruct] = []
    embed_batch_texts: list[str] = []
    embed_batch_records: list[dict] = []

    for record in _stream_jsonl(file_path):
        total_processed += 1
        
        # Strategy 1: Use pre-computed vector if available
        vector = record.get("vector")
        if isinstance(vector, list) and len(vector) > 0:
            point = _record_to_point(record, vector)
            if point:
                points_buffer.append(point)
        # Strategy 2: Fallback to local embedding if model is available
        elif model is not None:
            content = str(record.get("content") or record.get("cleaned_text") or record.get("embedding_text") or record.get("title") or record.get("doc_title") or "").strip()
            if content:
                embed_batch_texts.append(f"passage: {content}")
                embed_batch_records.append(record)
        # Strategy 3: Skip if no vector and no model
        else:
            skipped_no_vector += 1
            continue

        # Process embedding batch if full
        if len(embed_batch_texts) >= embed_batch_size:
            vectors = _embed_batch(model, embed_batch_texts)
            for r, vec in zip(embed_batch_records, vectors):
                point = _record_to_point(r, vec)
                if point:
                    points_buffer.append(point)
            embed_batch_texts = []
            embed_batch_records = []

        # Upsert batch if full
        if len(points_buffer) >= upsert_batch_size:
            batch = points_buffer[:upsert_batch_size]
            client.upsert(collection_name=collection_name, points=batch, wait=False)
            total_upserted += len(batch)
            points_buffer = points_buffer[upsert_batch_size:]
            if total_upserted % 5000 == 0:
                _log.info("Processed %d records, upserted %d points", total_processed, total_upserted)

    # Flush remaining embedding batch
    if embed_batch_texts and model is not None:
        vectors = _embed_batch(model, embed_batch_texts)
        for r, vec in zip(embed_batch_records, vectors):
            point = _record_to_point(r, vec)
            if point:
                points_buffer.append(point)

    # Flush remaining points buffer
    if points_buffer:
        client.upsert(collection_name=collection_name, points=points_buffer, wait=False)
        total_upserted += len(points_buffer)

    if skipped_no_vector > 0:
        _log.warning("Skipped %d records: no pre-computed vector and no embedding model available.", skipped_no_vector)
        
    _log.info("Total processed: %d, total upserted: %d", total_processed, total_upserted)
    return total_upserted


def _create_payload_indexes(client: QdrantClient, collection_name: str, is_graph: bool = False) -> None:
    """Create payload indexes for date fields and graph fields."""
    # Date / status indexes (for all chunk collections)
    for field, schema in (
        ("effective_date", PayloadSchemaType.DATETIME),
        ("expiry_date", PayloadSchemaType.DATETIME),
        ("status", PayloadSchemaType.KEYWORD),
    ):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema,
            )
            _log.info("Created %s index on '%s' for '%s'", schema.name, field, collection_name)
        except Exception as e:
            _log.warning("Index on '%s' may already exist or failed: %s", field, e)

    if is_graph:
        # Graph traversal indexes
        for field in (
            "doc_id", "chunk_id", "context_chunk_id",
            "parent_id", "prev_chunk_id", "next_chunk_id",
            "source_id", "target_id", "relation_type",
        ):
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                _log.info("Created KEYWORD index on '%s' for '%s'", field, collection_name)
            except Exception as e:
                _log.warning("Index on '%s' may already exist or failed: %s", field, e)


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, recreate: bool, distance: Distance = Distance.COSINE) -> None:
    if recreate:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            _log.info("Deleted existing collection '%s'", collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        _log.info("Created collection '%s' (size=%d, distance=%s)", collection_name, vector_size, distance.name)
        _create_payload_indexes(client, collection_name, is_graph=True)
    else:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            _log.info("Created collection '%s' (size=%d, distance=%s)", collection_name, vector_size, distance.name)
            _create_payload_indexes(client, collection_name, is_graph=True)


def _upsert_edges_streaming(
    client: QdrantClient,
    collection_name: str,
    file_path: Path,
    vector_size: int,
    upsert_batch_size: int,
) -> int:
    """Upsert legal edges with dummy vectors (graph traversal, no semantic search)."""
    dummy_vector = [1e-6] * vector_size
    total_upserted = 0
    points_buffer: list[PointStruct] = []

    for record in _stream_jsonl(file_path):
        source_id = record.get("source_id")
        target_id = record.get("target_id")
        relation = record.get("relation_type")
        if source_id is None or target_id is None or relation is None:
            continue
        # Deterministic unique edge ID
        edge_id_raw = f"{source_id}__{relation}__{target_id}"
        point_id = _to_qdrant_id(edge_id_raw)
        payload = {
            "source_id": str(source_id),
            "target_id": str(target_id),
            "relation_type": str(relation),
            "confidence": record.get("confidence", 1.0),
        }
        points_buffer.append(PointStruct(id=point_id, vector=dummy_vector, payload=payload))

        if len(points_buffer) >= upsert_batch_size:
            batch = points_buffer[:upsert_batch_size]
            client.upsert(collection_name=collection_name, points=batch, wait=False)
            total_upserted += len(batch)
            points_buffer = points_buffer[upsert_batch_size:]
            if total_upserted % 5000 == 0:
                _log.info("Upserted %d edges", total_upserted)

    if points_buffer:
        client.upsert(collection_name=collection_name, points=points_buffer, wait=False)
        total_upserted += len(points_buffer)

    _log.info("Total edges upserted: %d", total_upserted)
    return total_upserted
    """Peek into the file to get the vector size if pre-computed vectors exist."""
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    vec = record.get("vector")
                    if isinstance(vec, list) and len(vec) > 0:
                        return len(vec)
                except json.JSONDecodeError:
                    pass
                break
    return None

def _get_vector_size_from_file(file_path: Path) -> int | None:
    """Peek into the file to get the vector size if pre-computed vectors exist."""
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    vec = record.get("vector")
                    if isinstance(vec, list) and len(vec) > 0:
                        return len(vec)
                except json.JSONDecodeError:
                    pass
                break
    return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Embed and upsert documents/articles/chunks/context/edges to Qdrant.")
    parser.add_argument("--type", choices=["docs", "articles", "chunks", "context", "edges", "all"], default="all", help="What to upsert")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant HTTP URL")
    parser.add_argument("--embed-batch-size", type=int, default=8, help="Embedding batch size (keep small for CPU)")
    parser.add_argument("--upsert-batch-size", type=int, default=512, help="Upsert batch size")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collections")
    parser.add_argument("--docs-file", default="data/processed/cleaned_documents_enriched.jsonl")
    parser.add_argument("--articles-file", default="data/processed/legal_nodes.jsonl")
    parser.add_argument("--chunks-file", default="data/processed/legal_chunks_embedded.jsonl")
    parser.add_argument("--context-chunks-file", default="data/processed/legal_context_chunks_embedded.jsonl")
    parser.add_argument("--edges-file", default="data/processed/legal_edges.jsonl")
    args = parser.parse_args()

    _ensure_cache()
    client = QdrantClient(url=args.qdrant_url)
    _log.info("Connected to Qdrant at %s", args.qdrant_url)

    # Smart model loading: only load if files don't have pre-computed vectors
    docs_path = Path(args.docs_file)
    articles_path = Path(args.articles_file)
    chunks_path = Path(args.chunks_file)
    context_chunks_path = Path(args.context_chunks_file)
    edges_path = Path(args.edges_file)

    files_to_check = []
    if args.type in ("docs", "all"):
        files_to_check.append(docs_path)
    if args.type in ("articles", "all"):
        files_to_check.append(articles_path)
    if args.type in ("chunks", "context", "all"):
        files_to_check.append(chunks_path)
        files_to_check.append(context_chunks_path)

    vector_size = None
    for fp in files_to_check:
        sz = _get_vector_size_from_file(fp)
        if sz is not None:
            vector_size = sz
            break

    has_precomputed = vector_size is not None
    vector_size = vector_size or 1024  # fallback to 1024 for bge-m3

    model = None
    if not has_precomputed:
        _log.info("Loading embedding model BAAI/bge-m3 (no pre-computed vectors found)...")
        model = SentenceTransformer("BAAI/bge-m3", local_files_only=True)
        model.max_seq_length = 8192
        vector_size = model.get_sentence_embedding_dimension()
        _log.info("Model loaded. Vector size: %d", vector_size)
    else:
        _log.info("✅ Pre-computed vectors detected (size: %d). Skipping model load for instant startup!", vector_size)

    total_upserted = 0

    if args.type in ("docs", "all"):
        docs_path = Path(args.docs_file)
        if docs_path.exists():
            _log.info("=== Upserting documents ===")
            _ensure_collection(client, "legal_docs", vector_size, args.recreate)
            t0 = time.time()
            count = _upsert_streaming(client, "legal_docs", model, docs_path, args.embed_batch_size, args.upsert_batch_size)
            _log.info("Documents upserted: %d in %.1fs", count, time.time() - t0)
            total_upserted += count
        else:
            _log.warning("Documents file not found: %s", docs_path)

    if args.type in ("articles", "all"):
        articles_path = Path(args.articles_file)
        if articles_path.exists():
            _log.info("=== Upserting articles ===")
            _ensure_collection(client, "legal_articles", vector_size, args.recreate)
            t0 = time.time()
            count = _upsert_streaming(client, "legal_articles", model, articles_path, args.embed_batch_size, args.upsert_batch_size)
            _log.info("Articles upserted: %d in %.1fs", count, time.time() - t0)
            total_upserted += count
        else:
            _log.warning("Articles file not found: %s", articles_path)

    if args.type in ("chunks", "all"):
        if chunks_path.exists():
            _log.info("=== Upserting chunks (pre-computed vectors) ===")
            _ensure_collection(client, "legal_chunks", vector_size, args.recreate)
            t0 = time.time()
            count = _upsert_streaming(client, "legal_chunks", None, chunks_path, args.embed_batch_size, args.upsert_batch_size)
            _log.info("Chunks upserted: %d in %.1fs", count, time.time() - t0)
            total_upserted += count
        else:
            _log.warning("Chunks file not found: %s", chunks_path)

    if args.type in ("context", "all"):
        if context_chunks_path.exists():
            _log.info("=== Upserting context chunks (pre-computed vectors) ===")
            _ensure_collection(client, "legal_context_chunks", vector_size, args.recreate)
            t0 = time.time()
            count = _upsert_streaming(client, "legal_context_chunks", None, context_chunks_path, args.embed_batch_size, args.upsert_batch_size)
            _log.info("Context chunks upserted: %d in %.1fs", count, time.time() - t0)
            total_upserted += count
        else:
            _log.warning("Context chunks file not found: %s", context_chunks_path)

    if args.type in ("edges", "all"):
        if edges_path.exists():
            _log.info("=== Upserting edges (dummy vectors) ===")
            _ensure_collection(client, "legal_edges", vector_size, args.recreate)
            t0 = time.time()
            count = _upsert_edges_streaming(client, "legal_edges", edges_path, vector_size, args.upsert_batch_size)
            _log.info("Edges upserted: %d in %.1fs", count, time.time() - t0)
            total_upserted += count
        else:
            _log.warning("Edges file not found: %s", edges_path)

    _log.info("=== Done. Total upserted: %d ===", total_upserted)

    print(json.dumps({
        "event": "upsert_complete",
        "total_upserted": total_upserted,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
