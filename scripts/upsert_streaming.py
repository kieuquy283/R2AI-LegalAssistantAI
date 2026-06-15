from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Generator
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


def _distance_from_name(name: str) -> Distance:
    normalized = str(name or "cosine").strip().lower()
    if normalized == "dot":
        return Distance.DOT
    if normalized == "euclid":
        return Distance.EUCLID
    if normalized == "manhattan":
        return Distance.MANHATTAN
    return Distance.COSINE


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


def _record_to_point(record: dict[str, Any]) -> PointStruct | None:
    vector = list(record.get("vector") or record.get("embedding") or [])
    if not vector:
        return None
    raw_id = record.get("id")
    if raw_id is None:
        return None
    point_id = _to_qdrant_id(raw_id)
    payload = {
        k: v
        for k, v in record.items()
        if k not in ("vector", "embedding", "collection_name", "id")
    }
    return PointStruct(id=point_id, vector=vector, payload=payload)


def _stream_records(input_dir: Path, expected_collection: str) -> Generator[dict, None, None]:
    shard_files = sorted(input_dir.rglob("*.jsonl.gz"))
    if not shard_files:
        shard_files = sorted(input_dir.rglob("*.jsonl"))
    if not shard_files:
        _log.error("No .jsonl.gz or .jsonl files found under %s", input_dir)
        sys.exit(1)

    for shard_path in shard_files:
        open_fn = gzip.open if str(shard_path).endswith(".gz") else open
        records_in_shard = 0
        try:
            with open_fn(shard_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        _log.warning("Malformed JSON in %s: %s", shard_path.name, exc)
                        continue
                    rec_col = record.get("collection_name") or ""
                    if rec_col and rec_col != expected_collection:
                        continue
                    vector = record.get("vector") or record.get("embedding") or []
                    if not vector or not isinstance(vector, (list, tuple)):
                        continue
                    yield record
                    records_in_shard += 1
        except Exception as exc:
            _log.warning("Error reading %s: %s", shard_path, exc)
            continue
        _log.info("Shard %s: %d records", shard_path.name, records_in_shard)


def _infer_vector_size(input_dir: Path, expected_collection: str) -> int:
    for record in _stream_records(input_dir, expected_collection):
        vector = record.get("vector") or record.get("embedding") or []
        if vector and isinstance(vector, (list, tuple)) and len(vector) > 0:
            return len(vector)
    _log.error("Could not infer vector size from any record")
    sys.exit(1)


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, recreate: bool) -> None:
    if recreate:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            _log.info("Deleted existing collection '%s'", collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        _log.info("Created collection '%s' (size=%d, distance=Cosine)", collection_name, vector_size)
    else:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            _log.info("Created collection '%s' (size=%d, distance=Cosine)", collection_name, vector_size)


def _upsert_streaming(
    client: QdrantClient,
    collection_name: str,
    input_dir: Path,
    expected_collection: str,
    batch_size: int,
) -> int:
    total_upserted = 0
    batch: list[PointStruct] = []
    total_read = 0

    for record in _stream_records(input_dir, expected_collection):
        total_read += 1
        point = _record_to_point(record)
        if point is None:
            continue
        batch.append(point)
        if len(batch) >= batch_size:
            client.upsert(collection_name=collection_name, points=batch, wait=True)
            total_upserted += len(batch)
            _log.info("Upserted %d points (total=%d)", len(batch), total_upserted)
            batch = []

    if batch:
        client.upsert(collection_name=collection_name, points=batch, wait=True)
        total_upserted += len(batch)
        _log.info("Upserted final %d points (total=%d)", len(batch), total_upserted)

    _log.info("Total records read: %d, total upserted: %d", total_read, total_upserted)
    return total_upserted


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Upsert exported vector shards (JSONL.GZ) into Qdrant with streaming to avoid OOM."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing *.jsonl.gz shards")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant HTTP URL")
    parser.add_argument("--collection-name", required=True, help="Target Qdrant collection")
    parser.add_argument("--batch-size", type=int, default=128, help="Upsert batch size")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection before upsert")
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url)
    _log.info("Connected to Qdrant at %s", args.qdrant_url)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        _log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    _log.info("Inferring vector size from first record...")
    vector_size = _infer_vector_size(input_dir, args.collection_name)
    _log.info("Inferred vector size: %d", vector_size)

    _ensure_collection(client, args.collection_name, vector_size, args.recreate)

    total_upserted = _upsert_streaming(
        client=client,
        collection_name=args.collection_name,
        input_dir=input_dir,
        expected_collection=args.collection_name,
        batch_size=args.batch_size,
    )

    final_count = int(client.count(collection_name=args.collection_name, exact=True).count)
    _log.info("Upsert complete: %d records upserted, final Qdrant count=%d", total_upserted, final_count)

    print(
        json.dumps(
            {
                "event": "upsert_complete",
                "collection": args.collection_name,
                "inferred_vector_size": vector_size,
                "records_upserted": total_upserted,
                "final_count": final_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
