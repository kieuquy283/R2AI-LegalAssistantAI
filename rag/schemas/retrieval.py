from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(slots=True)
class RetrievedDoc:
    text: str
    metadata: Dict[str, Any]


@dataclass(slots=True)
class RetrievalMetrics:
    hit: int
    recall: float
    mrr: float


@dataclass(slots=True)
class RetrievalRun:
    query: str
    retrieved_cids: List[Any]
    metrics: RetrievalMetrics
