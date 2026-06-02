from __future__ import annotations

from typing import List

from ..fusion import (
    reciprocal_rank_fusion,
    weighted_fusion,
)
from .query_analyzer import (
    QueryAnalyzer,
)
from .retrieval_policy import (
    AdaptiveRetrievalPolicy,
)
from ..schemas import RetrievalResult
from ..utils import (
    deduplicate_results,
)


class AdaptiveHybridRetriever:

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
    ) -> None:

        self.dense_retriever = (
            dense_retriever
        )

        self.sparse_retriever = (
            sparse_retriever
        )

        self.query_analyzer = (
            QueryAnalyzer()
        )

        self.policy_builder = (
            AdaptiveRetrievalPolicy()
        )

    def retrieve(
        self,
        query: str,
    ) -> List[RetrievalResult]:

        # ============================================
        # Analyze query
        # ============================================

        features = (
            self.query_analyzer.analyze(
                query
            )
        )

        # ============================================
        # Build policy
        # ============================================

        policy = (
            self.policy_builder.build_policy(
                features
            )
        )

        # ============================================
        # Retrieve
        # ============================================

        dense_results = (
            self.dense_retriever.retrieve(
                query=query,
                top_k=policy.candidate_k,
            )
        )

        sparse_results = (
            self.sparse_retriever.retrieve(
                query=query,
                top_k=policy.candidate_k,
            )
        )

        # ============================================
        # Fusion
        # ============================================

        if (
            policy.fusion_type
            == "weighted"
        ):

            results = (
                weighted_fusion(
                    dense_results,
                    sparse_results,
                    alpha=policy.alpha,
                )
            )

        else:

            results = (
                reciprocal_rank_fusion(
                    dense_results,
                    sparse_results,
                )
            )

        # ============================================
        # Deduplicate
        # ============================================

        results = deduplicate_results(
            results
        )

        return results[
            : policy.top_k
        ]
