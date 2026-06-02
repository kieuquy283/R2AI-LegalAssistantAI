from __future__ import annotations

import statistics
from typing import List


class ContextSelector:
    """
    Adaptive context selector for conversational RAG.

    Pipeline:
        reranked candidates
            ↓
        score distribution analysis
            ↓
        adaptive thresholding
            ↓
        duplicate filtering
            ↓
        token budget filtering
            ↓
        final contexts

    Behavior:
        - if only few contexts are highly relevant:
            return few

        - if many contexts are highly relevant:
            return many
    """

    def __init__(
        self,
        min_contexts: int = 1,
        max_contexts: int = 12,
        min_score: float = 0.55,
        relative_threshold: float = 0.8,
        score_alpha: float = 0.15,
        max_total_tokens: int = 1200,
        use_token_budget: bool = True,
    ) -> None:

        self.min_contexts = (
            min_contexts
        )

        self.max_contexts = (
            max_contexts
        )

        self.min_score = (
            min_score
        )

        self.relative_threshold = (
            relative_threshold
        )

        self.score_alpha = (
            score_alpha
        )

        self.max_total_tokens = (
            max_total_tokens
        )

        self.use_token_budget = (
            use_token_budget
        )

    # =====================================================
    # Main API
    # =====================================================

    def select(
        self,
        reranked_results,
    ):

        if not reranked_results:
            return []

        # ================================================
        # Deduplicate
        # ================================================

        reranked_results = (
            self._deduplicate(
                reranked_results
            )
        )

        # ================================================
        # Compute final scores
        # ================================================

        scored_results = []

        for result in reranked_results:

            retrieval_score = float(

                getattr(
                    result,
                    "retrieval_score",
                    0.0,
                )
            )

            rerank_score = float(

                getattr(
                    result,
                    "rerank_score",
                    0.0,
                )
            )

            final_score = (

                self.score_alpha
                * retrieval_score

                +

                (1 - self.score_alpha)
                * rerank_score
            )

            result.final_score = (
                final_score
            )

            scored_results.append(
                result
            )

        # ================================================
        # Sort by final score
        # ================================================

        scored_results = sorted(

            scored_results,

            key=lambda x: (
                x.final_score
            ),

            reverse=True,
        )

        # ================================================
        # Score distribution
        # ================================================

        scores = [

            result.final_score

            for result
            in scored_results
        ]

        top_score = max(scores)

        mean_score = statistics.mean(
            scores
        )

        std_score = (

            statistics.pstdev(
                scores
            )

            if len(scores) > 1

            else 0.0
        )

        threshold = (
            self._dynamic_threshold(

                top_score=top_score,

                mean_score=mean_score,

                std_score=std_score,
            )
        )

        # ================================================
        # Adaptive selection
        # ================================================

        selected = []

        total_tokens = 0

        for result in scored_results:

            score = (
                result.final_score
            )

            # ============================================
            # Absolute threshold
            # ============================================

            if score < self.min_score:
                continue

            # ============================================
            # Dynamic threshold
            # ============================================

            if score < threshold:
                continue

            # ============================================
            # Relative threshold
            # ============================================

            relative_score = (
                score
                /
                max(top_score, 1e-8)
            )

            if (
                relative_score
                < self.relative_threshold
            ):
                continue

            # ============================================
            # Token budget
            # ============================================

            token_count = (
                self._estimate_tokens(
                    result.text
                )
            )

            if (
                self.use_token_budget
            ):

                if (

                    total_tokens
                    + token_count

                    >

                    self.max_total_tokens
                ):
                    continue

            total_tokens += (
                token_count
            )

            selected.append(result)

            # ============================================
            # Max contexts
            # ============================================

            if (
                len(selected)
                >= self.max_contexts
            ):
                break

        # ================================================
        # Ensure minimum contexts
        # ================================================

        if (
            len(selected)
            < self.min_contexts
        ):

            selected = scored_results[
                : self.min_contexts
            ]

        return selected

    # =====================================================
    # Dynamic Threshold
    # =====================================================

    def _dynamic_threshold(
        self,
        top_score: float,
        mean_score: float,
        std_score: float,
    ) -> float:

        # ================================================
        # Very confident retrieval
        # ================================================

        if top_score >= 0.9:

            return max(

                0.75,

                top_score
                - (
                    std_score
                    * 0.5
                ),
            )

        # ================================================
        # Medium confidence
        # ================================================

        if top_score >= 0.8:

            return max(

                0.65,

                mean_score
                + (
                    std_score
                    * 0.3
                ),
            )

        # ================================================
        # Lower confidence
        # ================================================

        return max(

            self.min_score,

            mean_score,
        )

    # =====================================================
    # Deduplication
    # =====================================================

    def _deduplicate(
        self,
        results,
    ):

        unique = {}

        for result in results:

            chunk_id = getattr(
                result,
                "chunk_id",
                None,
            )

            if chunk_id is None:
                continue

            existing = unique.get(
                chunk_id
            )

            if existing is None:

                unique[
                    chunk_id
                ] = result

                continue

            existing_score = float(

                getattr(
                    existing,
                    "rerank_score",
                    0.0,
                )
            )

            current_score = float(

                getattr(
                    result,
                    "rerank_score",
                    0.0,
                )
            )

            if (
                current_score
                > existing_score
            ):

                unique[
                    chunk_id
                ] = result

        return list(
            unique.values()
        )

    # =====================================================
    # Token Estimation
    # =====================================================

    def _estimate_tokens(
        self,
        text: str,
    ) -> int:
        """
        Lightweight token estimation.

        Approx:
            1 token ≈ 0.75 words
        """

        if not text:
            return 0

        word_count = len(
            text.split()
        )

        return int(
            word_count * 1.3
        )

    # =====================================================
    # Diagnostics
    # =====================================================

    def diagnostics(
        self,
        selected_results,
    ):

        if not selected_results:

            return {

                "selected_count": 0,

                "total_tokens": 0,

                "top_score": 0.0,
            }

        scores = [

            result.final_score

            for result
            in selected_results
        ]

        total_tokens = sum(

            self._estimate_tokens(
                result.text
            )

            for result
            in selected_results
        )

        return {

            "selected_count": len(
                selected_results
            ),

            "total_tokens": (
                total_tokens
            ),

            "top_score": round(
                max(scores),
                4,
            ),

            "mean_score": round(
                statistics.mean(
                    scores
                ),
                4,
            ),
        }