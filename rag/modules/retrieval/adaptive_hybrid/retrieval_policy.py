from __future__ import annotations

from dataclasses import dataclass

from .query_analyzer import (
    QueryFeatures,
)


@dataclass
class RetrievalPolicy:

    alpha: float

    top_k: int

    candidate_k: int

    fusion_type: str


class AdaptiveRetrievalPolicy:

    def build_policy(
        self,
        features: QueryFeatures,
    ) -> RetrievalPolicy:

        # ============================================
        # Default
        # ============================================

        alpha = 0.7

        top_k = 5

        candidate_k = 15

        fusion_type = "rrf"

        # ============================================
        # Exact keyword queries
        # ============================================

        if (
            features.has_exact_pattern
            or features.has_numbers
            or features.has_acronym
        ):

            alpha = 0.3

        # ============================================
        # Conversational / semantic
        # ============================================

        if features.is_follow_up:

            alpha = 0.8

            top_k = 8

            candidate_k = 20

        # ============================================
        # Long semantic query
        # ============================================

        if (
            features.query_length
            >= 12
        ):

            alpha = 0.8

        return RetrievalPolicy(

            alpha=alpha,

            top_k=top_k,

            candidate_k=candidate_k,

            fusion_type=fusion_type,
        )