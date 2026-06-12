from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.config.runtime import RetrievalRuntimeConfig, get_retrieval_runtime_config


def _distance_from_name(name: str) -> Distance:
    normalized = str(name or "cosine").strip().lower()
    if normalized == "dot":
        return Distance.DOT
    if normalized == "euclid":
        return Distance.EUCLID
    if normalized == "manhattan":
        return Distance.MANHATTAN
    return Distance.COSINE


def _stable_point_id(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & 0x7FFFFFFFFFFFFFFF


@dataclass(frozen=True)
class QdrantCollectionSpec:
    name: str
    vector_size: int
    distance: Distance


class QdrantStore:
    def __init__(self, client: QdrantClient | None = None, config: RetrievalRuntimeConfig | None = None) -> None:
        import logging
        self.config = config or get_retrieval_runtime_config()
        _log = logging.getLogger(__name__)
        if client is not None:
            self.client = client
            self.mode = "provided"
            _log.info("QdrantStore using provided client")
        elif self.config.qdrant_url:
            self.mode = "url"
            _log.info("QdrantStore using url: %s", self.config.qdrant_url)
            self.client = QdrantClient(url=self.config.qdrant_url, api_key=self.config.qdrant_api_key or None)
        elif self.config.qdrant_path:
            self.mode = "path"
            _log.info("QdrantStore using local path: %s", self.config.qdrant_path)
            self.client = QdrantClient(path=self.config.qdrant_path)
        else:
            self.mode = "host:port"
            _log.info(
                "QdrantStore using host:port %s:%s",
                self.config.qdrant_host,
                self.config.qdrant_port,
            )
            self.client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                grpc_port=self.config.qdrant_grpc_port,
                api_key=self.config.qdrant_api_key or None,
                prefer_grpc=False,
            )

    def collection_spec(self, name: str) -> QdrantCollectionSpec:
        return QdrantCollectionSpec(
            name=name,
            vector_size=self.config.qdrant_vector_size,
            distance=_distance_from_name(self.config.qdrant_distance),
        )

    def recreate_collection(self, name: str) -> None:
        spec = self.collection_spec(name)
        if self.client.collection_exists(spec.name):
            self.client.delete_collection(spec.name)
        self.client.create_collection(
            collection_name=spec.name,
            vectors_config=VectorParams(size=spec.vector_size, distance=spec.distance),
        )

    def ensure_collection(self, name: str) -> None:
        spec = self.collection_spec(name)
        if self.client.collection_exists(spec.name):
            return
        self.client.create_collection(
            collection_name=spec.name,
            vectors_config=VectorParams(size=spec.vector_size, distance=spec.distance),
        )

    def upsert_rows(
        self,
        *,
        collection_name: str,
        rows: Iterable[dict[str, Any]],
        vector_key: str,
        id_key: str,
    ) -> int:
        points: list[PointStruct] = []
        for row in rows:
            vector = list(row.get(vector_key) or [])
            row_id = str(row.get(id_key) or "").strip()
            if not row_id or not vector:
                continue
            payload = {key: value for key, value in row.items() if key != vector_key}
            points.append(
                PointStruct(
                    id=_stable_point_id(f"{collection_name}:{row_id}"),
                    vector=vector,
                    payload=payload,
                )
            )
        if points:
            self.client.upsert(collection_name=collection_name, points=points, wait=True)
        return len(points)

    def count(self, collection_name: str) -> int:
        return int(self.client.count(collection_name=collection_name, exact=True).count)
