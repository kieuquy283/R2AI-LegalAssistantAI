from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RetrievalResult:
    """
    Unified retrieval result schema.
    """

    chunk_id: str
    text: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_rank: int = -1
    rerank_score: float | None = None
    final_score: float | None = None
    raw_score: float | None = None
    normalized_score: float | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    sources: List[str] | None = None
    retriever_name: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": float(self.score),
            "source": self.source,
            "metadata": self.metadata,
            "retrieval_rank": int(self.retrieval_rank),
            "rerank_score": None if self.rerank_score is None else float(self.rerank_score),
            "final_score": None if self.final_score is None else float(self.final_score),
            "raw_score": None if self.raw_score is None else float(self.raw_score),
            "normalized_score": None if self.normalized_score is None else float(self.normalized_score),
            "dense_score": None if self.dense_score is None else float(self.dense_score),
            "sparse_score": None if self.sparse_score is None else float(self.sparse_score),
            "sources": list(self.sources) if self.sources is not None else None,
            "retriever_name": self.retriever_name,
        }
