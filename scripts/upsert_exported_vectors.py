from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
from pathlib import Path
from uuid import UUID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

_KNOWN_SMOKE_STORES = [
    "data/qdrant_question_targeted_smoke_v2",
    "data/qdrant_question_targeted_tax_smoke_run",
    "data/qdrant_targeted_curated_smoke_v2",
]


def _warn_if_qdrant_path_points_to_smoke(qdrant_path: str) -> None:
    resolved = str(Path(qdrant_path).resolve())
    for smoke in _KNOWN_SMOKE_STORES:
        if smoke in resolved:
            _log.warning(
                "QDRANT_PATH points to a known smoke/test store: %s\n"
                "  If you intend to use Docker Qdrant, unset QDRANT_PATH and set QDRANT_URL instead.",
                resolved,
            )


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
    # Hash string to stable integer
    digest = hashlib.sha256(str(raw_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & 0x7FFFFFFFFFFFFFFF


def _read_gz_records(input_dir: Path, expected_collection: str) -> list[dict]:
    records: list[dict] = []
    shard_files = sorted(input_dir.rglob("*.jsonl.gz"))
    if not shard_files:
        shard_files = sorted(input_dir.rglob("*.jsonl"))
    if not shard_files:
        _log.error("No .jsonl.gz or .jsonl files found under %s", input_dir)
        sys.exit(1)

    for shard_path in shard_files:
        open_fn = gzip.open if str(shard_path).endswith(".gz") else open
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
                    # Validate collection name if present
                    rec_col = record.get("collection_name") or ""
                    if rec_col and rec_col != expected_collection:
                        _log.warning(
                            "Collection mismatch in %s: expected '%s', got '%s' — skipping",
                            shard_path.name,
                            expected_collection,
                            rec_col,
                        )
                        continue
                    vector = record.get("vector") or record.get("embedding") or []
                    if not vector or not isinstance(vector, (list, tuple)):
                        _log.warning("Missing/empty vector in %s (id=%s)", shard_path.name, record.get("id", "?"))
                        continue
                    if all(v == 0.0 for v in vector):
                        _log.warning("Zero vector in %s (id=%s)", shard_path.name, record.get("id", "?"))
                        records.append(record)
                    else:
                        records.append(record)
        except Exception as exc:
            _log.warning("Error reading %s: %s", shard_path, exc)
            continue

    _log.info("Read %d shard files, %d total records", len(shard_files), len(records))
    return records


def _infer_vector_size(records: list[dict]) -> int:
    for record in records:
        vector = record.get("vector") or record.get("embedding") or []
        if vector and isinstance(vector, (list, tuple)) and len(vector) > 0:
            return len(vector)
    _log.error("Could not infer vector size from any record")
    sys.exit(1)


def _upsert_records(
    client: QdrantClient,
    collection_name: str,
    records: list[dict],
    batch_size: int,
    recreate: bool,
) -> int:
    vector_size = _infer_vector_size(records)
    _log.info("Inferred vector size: %d", vector_size)

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

    total_upserted = 0
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]
        points: list[PointStruct] = []
        for record in batch:
            vector = list(record.get("vector") or record.get("embedding") or [])
            raw_id = record.get("id")
            if raw_id is None:
                continue
            point_id = _to_qdrant_id(raw_id)
            payload = {
                k: v
                for k, v in record.items()
                if k not in ("vector", "embedding", "collection_name", "id")
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        if points:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            total_upserted += len(points)

    return total_upserted


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Upsert exported vector shards (JSONL.GZ) into Qdrant."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing *.jsonl.gz shards")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant HTTP URL")
    parser.add_argument("--collection-name", required=True, help="Target Qdrant collection")
    parser.add_argument("--batch-size", type=int, default=128, help="Upsert batch size")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection before upsert")
    args = parser.parse_args()

    _warn_if_qdrant_path_points_to_smoke(str(Path(args.input_dir)))

    client = QdrantClient(url=args.qdrant_url)
    _log.info("Connected to Qdrant at %s", args.qdrant_url)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        _log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    records = _read_gz_records(input_dir, args.collection_name)
    if not records:
        _log.error("No valid records found in %s", input_dir)
        sys.exit(1)

    _log.info("Input records: %d", len(records))

    total_upserted = _upsert_records(
        client=client,
        collection_name=args.collection_name,
        records=records,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )

    # Verify count
    final_count = int(client.count(collection_name=args.collection_name, exact=True).count)
    _log.info(
        "Upsert complete: %d records upserted, final Qdrant count=%d",
        total_upserted,
        final_count,
    )

    print(
        json.dumps(
            {
                "event": "upsert_complete",
                "collection": args.collection_name,
                "shard_files": len(list(Path(args.input_dir).rglob("*.jsonl.gz"))),
                "inferred_vector_size": _infer_vector_size(records),
                "records_read": len(records),
                "records_upserted": total_upserted,
                "final_count": final_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
