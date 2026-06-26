from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import unicodedata
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from src.ingestion.common import read_jsonl
from rag.modules.retrieval.utils import tokenize_for_bm25
import portalocker

_BM25_K1 = float(os.getenv("R2AI_BM25_K1", "1.5"))
_BM25_B = float(os.getenv("R2AI_BM25_B", "0.75"))


# Try orjson for faster JSON parsing
try:
    import orjson
    _has_orjson = True
except Exception:
    _has_orjson = False


def _fast_jsonl_read(path: Path) -> list[dict[str, Any]]:
    """Read JSONL with orjson if available for faster parsing."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        if _has_orjson:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(orjson.loads(line))
                except Exception:
                    rows.append(json.loads(line))
        else:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d").replace("Đ", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _file_hash(path: Path) -> str:
    """Return a fast hash of file size + mtime for cache invalidation."""
    stat = path.stat()
    return hashlib.md5(f"{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()[:16]


class _FastBM25:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]], *, k1: float | None = None, b: float | None = None) -> None:
        self.corpus_tokens = [list(tokens) for tokens in corpus_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.k1 = k1 if k1 is not None else _BM25_K1
        self.b = b if b is not None else _BM25_B
        self.doc_freq: dict[str, int] = {}
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, tokens in enumerate(self.corpus_tokens):
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, freq in counts.items():
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
                self.postings[token].append((doc_idx, freq))

    def get_scores(self, query_tokens: Sequence[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        total_docs = getattr(self, 'num_docs', len(getattr(self, 'corpus_tokens', [])))
        total_docs = max(total_docs, 1)
        for token in query_tokens:
            postings = self.postings.get(token)
            if not postings:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
            for doc_idx, freq in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avgdl, 1e-8)))
                scores[doc_idx] += idf * ((freq * (self.k1 + 1)) / max(denom, 1e-8))
        return scores


class BM25Retriever:
    def __init__(self, *, chunks_path: str | Path | None = None) -> None:
        self.chunks_path = Path(chunks_path or self._default_chunks_path())
        self._rows: list[dict[str, Any]] | None = None
        self._bm25: _FastBM25 | None = None
        self._cache_path = Path(os.getenv("R2AI_CACHE_DIR", "data/cache")) / "bm25_cache.pkl"
        self._cache_loaded = False
        self._sqlite_db_path = Path(os.getenv("R2AI_CACHE_DIR", "data/cache")) / "chunks.db"
        self._sqlite_conn: Any | None = None
        self._use_sqlite = self._sqlite_db_path.exists()
        if self._use_sqlite:
            print(f"[BM25] SQLite available: {self._sqlite_db_path}")
        else:
            print(f"[BM25] SQLite not found, will use JSONL: {self._sqlite_db_path}")

    def _ensure_cache_dir(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_sqlite_rows(self, indices: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch rows by index from SQLite. Returns dict mapping index -> row."""
        if not self._sqlite_conn:
            import sqlite3
            self._sqlite_conn = sqlite3.connect(self._sqlite_db_path)
            self._sqlite_conn.row_factory = sqlite3.Row
        
        # Batch query for performance
        placeholders = ",".join("?" for _ in indices)
        cursor = self._sqlite_conn.execute(
            f"SELECT * FROM chunks WHERE row_idx IN ({placeholders})",
            indices,
        )
        rows = {}
        for row in cursor:
            row_dict = dict(row)
            row_idx = row_dict["row_idx"]
            # Parse metadata JSON
            metadata = row_dict.get("metadata")
            if metadata:
                try:
                    row_dict["metadata"] = json.loads(metadata)
                except Exception:
                    row_dict["metadata"] = {}
            rows[row_idx] = row_dict
        return rows

    def _try_load_cache(self) -> bool:
        """Try to load BM25 index from disk cache. Returns True if successful."""
        if not self._cache_path.exists():
            return False
        if not self.chunks_path.exists():
            return False
        try:
            current_hash = _file_hash(self.chunks_path)
            with open(self._cache_path, "rb") as f:
                portalocker.lock(f, portalocker.LOCK_SH)
                data = pickle.loads(zlib.decompress(f.read()))
                portalocker.unlock(f)
            if data.get("file_hash") != current_hash:
                print(f"[BM25] Cache stale (hash mismatch), rebuilding...")
                return False
            # Only load index data, not full rows (rows will be loaded from SQLite on demand)
            self._bm25 = _FastBM25.__new__(_FastBM25)
            self._bm25.num_docs = data["num_docs"]
            self._bm25.doc_lengths = data["doc_lengths"]
            self._bm25.avgdl = data["avgdl"]
            self._bm25.k1 = data.get("k1", _BM25_K1)  # fallback for legacy cache
            self._bm25.b = data.get("b", _BM25_B)
            self._bm25.doc_freq = data["doc_freq"]
            self._bm25.postings = defaultdict(list, data["postings"])
            self._cache_loaded = True
            print(f"[BM25] Loaded cached index: {self._bm25.num_docs} docs, {len(self._bm25.doc_freq)} terms (k1={self._bm25.k1}, b={self._bm25.b})")
            return True
        except Exception as exc:
            print(f"[BM25] Cache load failed: {exc}. Will rebuild.")
            return False

    def _save_cache(self) -> None:
        """Save BM25 index to disk cache."""
        if self._bm25 is None:
            return
        self._ensure_cache_dir()
        try:
            data = {
                "file_hash": _file_hash(self.chunks_path),
                "num_docs": len(self._bm25.corpus_tokens) if hasattr(self._bm25, 'corpus_tokens') else getattr(self._bm25, 'num_docs', 0),
                "doc_lengths": self._bm25.doc_lengths,
                "avgdl": self._bm25.avgdl,
                "k1": self._bm25.k1,
                "b": self._bm25.b,
                "doc_freq": self._bm25.doc_freq,
                "postings": dict(self._bm25.postings),
                "_bm25_params_version": 2,
            }
            compressed = zlib.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            with open(self._cache_path, "wb") as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                f.write(compressed)
                portalocker.unlock(f)
            cache_size_mb = len(compressed) / 1024 / 1024
            print(f"[BM25] Saved cached index ({cache_size_mb:.1f} MB) to {self._cache_path}")
        except Exception as exc:
            print(f"[BM25] Cache save failed: {exc}")

    def _build_bm25_from_rows(self, rows: list[dict]) -> _FastBM25:
        from rag.modules.retrieval.utils import tokenize_for_bm25
        corpus_tokens = []
        for i, row in enumerate(rows):
            if i % 50000 == 0:
                print(f"[BM25] Tokenizing {i}/{len(rows)}...", flush=True)
            corpus_tokens.append(tokenize_for_bm25(_normalize_text(self._combined_text(row))))
        return _FastBM25(corpus_tokens)

    def _build_bm25_from_sqlite(self) -> _FastBM25:
        print(f"[BM25] Building index from SQLite ({self._sqlite_db_path})...", flush=True)
        import sqlite3
        conn = sqlite3.connect(self._sqlite_db_path)
        conn.row_factory = sqlite3.Row
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        rows = []
        batch_size = 10000
        offset = 0
        while offset < count:
            cursor = conn.execute(
                f"SELECT * FROM chunks ORDER BY row_idx LIMIT {batch_size} OFFSET {offset}"
            )
            for row in cursor:
                row_dict = dict(row)
                metadata = row_dict.get("metadata")
                if metadata:
                    try:
                        row_dict["metadata"] = json.loads(metadata)
                    except Exception:
                        row_dict["metadata"] = {}
                rows.append(row_dict)
            offset += batch_size
            print(f"[BM25] SQLite rows loaded: {len(rows)}/{count}", flush=True)
        conn.close()
        return self._build_bm25_from_rows(rows)

    def preload(self) -> None:
        """Trigger eager loading of BM25 index. Call at startup for pre-warm."""
        import time
        t0 = time.perf_counter()
        
        if self._cache_loaded:
            return
        
        if not self._cache_loaded:
            self._try_load_cache()
        
        if self._bm25 is None and self._use_sqlite and self._sqlite_db_path.exists():
            self._bm25 = self._build_bm25_from_sqlite()
            self._save_cache()
        elif self._bm25 is None:
            print(f"[BM25] Building index from {self.chunks_path}...", flush=True)
            _ = self.bm25
            self._save_cache()
        
        t1 = time.perf_counter()
        print(f"[BM25] Preload: {t1-t0:.2f}s")

    @property
    def rows(self) -> list[dict[str, Any]]:
        # Fallback to JSONL if SQLite not available
        if self._rows is None:
            self._rows = _fast_jsonl_read(self.chunks_path)
        return self._rows

    @property
    def bm25(self) -> _FastBM25:
        if self._bm25 is None:
            corpus_tokens = [tokenize_for_bm25(_normalize_text(self._combined_text(row))) for row in self.rows]
            self._bm25 = _FastBM25(corpus_tokens)
        return self._bm25

    @staticmethod
    def _default_chunks_path() -> Path:
        overridden = os.getenv("R2AI_CHUNKS_PATH", "").strip()
        if overridden:
            return Path(overridden)
        merged = Path("data/processed/merged_chunks.jsonl")
        if merged.exists():
            return merged
        return Path("data/processed/chunks.jsonl")

    @staticmethod
    def _combined_text(row: dict[str, Any]) -> str:
        parts = [
            row.get("doc_title"),
            row.get("doc_number"),
            row.get("citation"),
            row.get("article"),
            row.get("clause"),
            row.get("content"),
            row.get("domain"),
        ]
        return "\n".join(str(part or "").strip() for part in parts if str(part or "").strip())

    @staticmethod
    def _allowed_domain(row: dict[str, Any], preferred_domains: Sequence[str] | None) -> bool:
        return True

    @staticmethod
    def _normalize_scores(values: list[float]) -> list[float]:
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if maximum == minimum:
            return [1.0 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def search(self, query: str, *, top_k: int, preferred_domains: Sequence[str] | None = None) -> list[dict[str, Any]]:
        # Early exit for empty or whitespace-only query
        if not query or not query.strip():
            return []
        
        query_tokens = tokenize_for_bm25(_normalize_text(query))
        if not query_tokens:
            return []
        
        raw_scores = self.bm25.get_scores(query_tokens)
        if not raw_scores:
            return []
        
        # Reduced multiplier from 3 to 2 for faster search
        ranked = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)[: max(top_k * 2, top_k)]
        normalized = self._normalize_scores([float(score) for _idx, score in ranked])
        
        # Get row indices we need
        needed_indices = [idx for (idx, _) in ranked]
        
        # Load rows efficiently: SQLite random access or fallback to JSONL
        if self._use_sqlite and self._cache_loaded:
            # Use SQLite for random access (only load needed rows)
            sqlite_rows = self._get_sqlite_rows(needed_indices)
        else:
            # Fallback: load all rows from JSONL
            sqlite_rows = None
        
        candidates: list[dict[str, Any]] = []
        for (idx, _score), normalized_score in zip(ranked, normalized, strict=True):
            if sqlite_rows is not None:
                row = sqlite_rows.get(idx)
                if row is None:
                    continue
            else:
                if idx >= len(self.rows):
                    continue
                row = dict(self.rows[idx])
            
            if not self._allowed_domain(row, preferred_domains):
                continue
            candidates.append(
                {
                    "candidate_id": f"chunk:{row.get('chunk_id')}",
                    "retrieval_level": "chunk",
                    "retrieval_method": "bm25",
                    "dense_score": 0.0,
                    "bm25_score": float(normalized_score),
                    "exact_score": 0.0,
                    "title_overlap": 0.0,
                    "lexical_overlap": 0.0,
                    "domain_match": 0.0,
                    "domain_score": 0.0,
                    "citation_match": 0.0,
                    "final_score": float(normalized_score),
                    "confidence": float(normalized_score),
                    "chunk_id": str(row.get("chunk_id") or ""),
                    "doc_id": str(row.get("doc_id") or ""),
                    "article_id": str(row.get("node_id") or ""),
                    "chunk_ref_id": str(row.get("chunk_id") or ""),
                    "doc_number": str(row.get("doc_number") or ""),
                    "doc_title": str(row.get("doc_title") or ""),
                    "article": str(row.get("article") or ""),
                    "clause": str(row.get("clause") or ""),
                    "citation": str(row.get("citation") or row.get("doc_title") or ""),
                    "domain": str(row.get("domain") or ""),
                    "source_url": str(row.get("source_url") or ""),
                    "content": str(row.get("content") or ""),
                    "source_dataset": str(row.get("source_dataset") or "local_corpus"),
                    "priority": int(row.get("priority") or 0),
                    "metadata": row,
                }
            )
            if len(candidates) >= top_k:
                break
        return candidates
