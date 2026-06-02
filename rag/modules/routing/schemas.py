from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class PipelineLevel(str, Enum):
    SIMPLE_DENSE = "simple_dense"
    REWRITE_DENSE = "rewrite_dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"
    FULL_OPTIMAL = "full_optimal"
    HYDE = "hyde"


@dataclass
class RouteDecision:
    level: PipelineLevel
    analyzer_signals: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    estimated_cost: str = "cheap"


@dataclass
class RetrievalConfidence:
    score: float
    doc_count: int
    unique_cid_count: int
    has_cid_metadata: bool
    non_empty_content_ratio: float
    duplicate_rate: float
    score_signal_available: bool
    top_score: float | None = None
    score_gap: float | None = None
    reasons: List[str] = field(default_factory=list)

