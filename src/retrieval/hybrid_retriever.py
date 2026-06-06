from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import faiss
import numpy as np

from rag.modules.retrieval.utils import tokenize_for_bm25
from rag.retrieval.vectorstore import get_embeddings
from src.ingestion.common import read_jsonl


@dataclass
class _SearchItem:
    chunk_id: str
    score: float
    retrieval_method: str
    content: str
    embedding_text: str
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "score": float(self.score),
            "retrieval_score": float(self.score),
            "retrieval_method": self.retrieval_method,
            "content": self.content,
            "embedding_text": self.embedding_text,
            "metadata": self.metadata,
        }


class _SimpleBM25:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]]) -> None:
        import math

        self._math = math
        self.corpus_tokens = [list(tokens) for tokens in corpus_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.k1 = 1.5
        self.b = 0.75
        self.doc_freq: Dict[str, int] = {}
        for tokens in self.corpus_tokens:
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
        scores: List[float] = []
        total_docs = max(len(self.corpus_tokens), 1)
        for tokens in self.corpus_tokens:
            token_counts: Dict[str, int] = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            doc_len = len(tokens)
            score = 0.0
            for token in query_tokens:
                freq = token_counts.get(token, 0)
                if freq == 0:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = self._math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
                denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avgdl, 1e-8)))
                score += idf * ((freq * (self.k1 + 1)) / max(denom, 1e-8))
            scores.append(score)
        return scores


class HybridRetriever:
    def __init__(
        self,
        *,
        faiss_index_path: str | Path = "data/indexes/faiss.index",
        metadata_path: str | Path = "data/indexes/chunk_metadata.json",
        chunks_path: str | Path = "data/processed/chunks.jsonl",
        bm25_corpus_path: str | Path = "data/indexes/bm25_corpus.json",
        embedding_model=None,
        alpha: float = 0.6,
    ) -> None:
        self.faiss_index_path = Path(faiss_index_path)
        self.metadata_path = Path(metadata_path)
        self.chunks_path = Path(chunks_path)
        self.bm25_corpus_path = Path(bm25_corpus_path)
        self.embedding_model = embedding_model or get_embeddings()
        self.alpha = float(alpha)

        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.faiss_index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing chunk metadata: {self.metadata_path}")
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Missing chunks file: {self.chunks_path}")

        self.index = faiss.read_index(str(self.faiss_index_path))
        self.metadata_rows = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.chunks = read_jsonl(self.chunks_path)
        self.chunk_by_id = {str(row["chunk_id"]): row for row in self.chunks}
        self.metadata_by_index = {int(row["index"]): row for row in self.metadata_rows}
        self._init_bm25()

    def _init_bm25(self) -> None:
        if self.bm25_corpus_path.exists():
            corpus_rows = json.loads(self.bm25_corpus_path.read_text(encoding="utf-8"))
            self.bm25_chunk_ids = [str(row["chunk_id"]) for row in corpus_rows]
            tokenized = [list(row.get("tokens") or []) for row in corpus_rows]
        else:
            self.bm25_chunk_ids = [str(row["chunk_id"]) for row in self.chunks]
            tokenized = [tokenize_for_bm25(str(row.get("embedding_text") or row.get("content") or "")) for row in self.chunks]

        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(tokenized)
        except ModuleNotFoundError:
            self.bm25 = _SimpleBM25(tokenized)

    def _normalize(self, scores: Sequence[float]) -> List[float]:
        values = [float(score) for score in scores]
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if maximum == minimum:
            return [1.0 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def _allowed_domain(self, metadata: Dict[str, object], domain: str | Sequence[str] | None) -> bool:
        if domain is None:
            return True
        allowed = {domain} if isinstance(domain, str) else set(domain)
        return str(metadata.get("domain")) in allowed

    def _make_item(self, chunk_id: str, score: float, method: str) -> _SearchItem | None:
        chunk = self.chunk_by_id.get(chunk_id)
        if not chunk:
            return None
        metadata = {
            "doc_id": chunk.get("doc_id"),
            "domain": chunk.get("domain"),
            "doc_title": chunk.get("doc_title"),
            "article": chunk.get("article"),
            "clause": chunk.get("clause"),
            "citation": chunk.get("citation"),
            "source_url": chunk.get("source_url"),
        }
        return _SearchItem(
            chunk_id=chunk_id,
            score=float(score),
            retrieval_method=method,
            content=str(chunk.get("content") or ""),
            embedding_text=str(chunk.get("embedding_text") or chunk.get("content") or ""),
            metadata=metadata,
        )

    def _dense_search(self, query: str, *, top_k: int, domain: str | Sequence[str] | None = None) -> List[_SearchItem]:
        query_vector = np.array([self.embedding_model.embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, max(top_k * 4, top_k))
        items: List[_SearchItem] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            metadata_row = self.metadata_by_index.get(int(index))
            if not metadata_row or not self._allowed_domain(metadata_row, domain):
                continue
            item = self._make_item(str(metadata_row["chunk_id"]), float(score), "dense")
            if item:
                items.append(item)
            if len(items) >= top_k:
                break
        return items

    def _sparse_search(self, query: str, *, top_k: int, domain: str | Sequence[str] | None = None) -> List[_SearchItem]:
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[: max(top_k * 4, top_k)]
        ranked_scores = [scores[idx] for idx in ranked_indices]
        normalized = self._normalize(ranked_scores)
        items: List[_SearchItem] = []
        for idx, score in zip(ranked_indices, normalized):
            chunk_id = self.bm25_chunk_ids[idx]
            chunk = self.chunk_by_id.get(chunk_id)
            if not chunk or not self._allowed_domain(chunk, domain):
                continue
            item = self._make_item(chunk_id, float(score), "sparse")
            if item:
                items.append(item)
            if len(items) >= top_k:
                break
        return items

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        domain: str | Sequence[str] | None = None,
    ) -> List[Dict[str, object]]:
        dense = self._dense_search(query, top_k=top_k, domain=domain)
        sparse = self._sparse_search(query, top_k=top_k, domain=domain)

        fused: Dict[str, Dict[str, object]] = {}
        for item in dense:
            fused[item.chunk_id] = item.to_dict()
            fused[item.chunk_id]["score"] = self.alpha * item.score
            fused[item.chunk_id]["retrieval_score"] = fused[item.chunk_id]["score"]
        for item in sparse:
            existing = fused.get(item.chunk_id)
            if existing is None:
                fused[item.chunk_id] = item.to_dict()
                fused[item.chunk_id]["score"] = (1.0 - self.alpha) * item.score
                fused[item.chunk_id]["retrieval_score"] = fused[item.chunk_id]["score"]
                continue
            existing["retrieval_method"] = "hybrid"
            existing["score"] = float(existing["score"]) + (1.0 - self.alpha) * item.score
            existing["retrieval_score"] = existing["score"]

        ranked = sorted(fused.values(), key=lambda row: float(row["score"]), reverse=True)
        return ranked[:top_k]


def _cli() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Search ingestion index with hybrid dense + BM25 retrieval.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--domain", action="append", default=None)
    args = parser.parse_args()

    retriever = HybridRetriever()
    results = retriever.search(args.query, top_k=args.top_k, domain=args.domain or None)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
