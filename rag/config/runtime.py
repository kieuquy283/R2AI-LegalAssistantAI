from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return default


def _get_text(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class RetrievalRuntimeConfig:
    retrieval_backend: str = field(default_factory=lambda: _get_text("RETRIEVAL_BACKEND", "qdrant").lower())
    qdrant_host: str = field(default_factory=lambda: _get_text("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: _get_int("QDRANT_PORT", 6333))
    qdrant_grpc_port: int = field(default_factory=lambda: _get_int("QDRANT_GRPC_PORT", 6334))
    qdrant_url: str = field(default_factory=lambda: _get_text("QDRANT_URL", ""))
    qdrant_path: str = field(default_factory=lambda: _get_text("QDRANT_PATH", ""))
    qdrant_api_key: str = field(default_factory=lambda: _get_text("QDRANT_API_KEY", ""))
    qdrant_collection_docs: str = field(default_factory=lambda: _get_text("QDRANT_COLLECTION_DOCS", "legal_docs"))
    qdrant_collection_articles: str = field(default_factory=lambda: _get_text("QDRANT_COLLECTION_ARTICLES", "legal_articles"))
    qdrant_collection_chunks: str = field(default_factory=lambda: _get_text("QDRANT_COLLECTION_CHUNKS", "legal_chunks"))
    qdrant_vector_size: int = field(default_factory=lambda: _get_int("QDRANT_VECTOR_SIZE", 1024))
    qdrant_distance: str = field(default_factory=lambda: _get_text("QDRANT_DISTANCE", "Cosine"))
    candidate_k_docs: int = field(default_factory=lambda: _get_int("CANDIDATE_K_DOCS", 50))
    candidate_k_articles: int = field(default_factory=lambda: _get_int("CANDIDATE_K_ARTICLES", 100))
    candidate_k_chunks: int = field(default_factory=lambda: _get_int("CANDIDATE_K_CHUNKS", 150))
    candidate_k_sparse: int = field(default_factory=lambda: _get_int("CANDIDATE_K_SPARSE", 150))
    candidate_k_title: int = field(default_factory=lambda: _get_int("CANDIDATE_K_TITLE", 50))
    rerank_top_n: int = field(default_factory=lambda: _get_int("RERANK_TOP_N", 50))
    min_contexts: int = field(default_factory=lambda: _get_int("MIN_CONTEXTS", 1))
    max_contexts: int = field(default_factory=lambda: _get_int("MAX_CONTEXTS", 8))
    max_docs: int = field(default_factory=lambda: _get_int("MAX_DOCS", 2))
    max_articles: int = field(default_factory=lambda: _get_int("MAX_ARTICLES", 3))
    absolute_score_threshold: float = field(default_factory=lambda: _get_float("ABSOLUTE_SCORE_THRESHOLD", 0.45))
    relative_score_threshold: float = field(default_factory=lambda: _get_float("RELATIVE_SCORE_THRESHOLD", 0.75))
    citation_score_threshold: float = field(default_factory=lambda: _get_float("CITATION_SCORE_THRESHOLD", 0.50))

    # HNSW index config (Qdrant)
    hnsw_ef_construction: int = field(default_factory=lambda: _get_int("HNSW_EF_CONSTRUCTION", 200))
    hnsw_m: int = field(default_factory=lambda: _get_int("HNSW_M", 16))
    hnsw_ef_search: int = field(default_factory=lambda: _get_int("HNSW_EF_SEARCH", 128))
    hnsw_on_disk: bool = field(default_factory=lambda: _get_text("HNSW_ON_DISK", "true").lower() == "true")
    hnsw_full_scan_threshold: int = field(default_factory=lambda: _get_int("HNSW_FULL_SCAN_THRESHOLD", 20000))


def get_retrieval_runtime_config() -> RetrievalRuntimeConfig:
    return RetrievalRuntimeConfig()
