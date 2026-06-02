from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RerankResult:
    """
    Unified reranker output schema.
    """

    chunk_id: str
    text: str
    retrieval_score: float
    rerank_score: float
    final_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_rank: int = -1
    rerank_rank: int = -1
    source: str = "reranker"
    raw_rerank_score: float | None = None
    normalized_rerank_score: float | None = None
    rank_delta: int | None = None
    reranker_name: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "retrieval_score": float(self.retrieval_score),
            "rerank_score": float(self.rerank_score),
            "final_score": float(self.final_score),
            "metadata": self.metadata,
            "retrieval_rank": int(self.retrieval_rank),
            "rerank_rank": int(self.rerank_rank),
            "source": self.source,
            "raw_rerank_score": None if self.raw_rerank_score is None else float(self.raw_rerank_score),
            "normalized_rerank_score": (
                None if self.normalized_rerank_score is None else float(self.normalized_rerank_score)
            ),
            "rank_delta": self.rank_delta,
            "reranker_name": self.reranker_name,
        }
