from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Sequence

from src.retrieval.reranker import Reranker as HeuristicReranker

LOGGER = logging.getLogger(__name__)


class HybridReranker:
    """Cascade reranker: heuristic first (fast) then cross-encoder (accurate)."""

    def __init__(
        self,
        *,
        heuristic_top_k: int = 50,
        max_contexts: int = 5,
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 8,
        max_length: int = 512,
        heuristic_weight: float = 0.3,
        cross_weight: float = 0.7,
        enable_cross_encoder: bool | None = None,
    ) -> None:
        self.heuristic = HeuristicReranker()
        self.heuristic_top_k = int(os.getenv("HYBRID_RERANKER_HEURISTIC_TOP_K", heuristic_top_k))
        self.max_contexts = int(max_contexts)
        self.cross_encoder_model = str(os.getenv("HYBRID_RERANKER_MODEL", cross_encoder_model))
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.heuristic_weight = float(heuristic_weight)
        self.cross_weight = float(os.getenv("HYBRID_RERANKER_CROSS_WEIGHT", cross_weight))
        self._cross_encoder: Any | None = None
        self._cross_encoder_available = False

        if enable_cross_encoder is None:
            enable_cross_encoder = os.getenv("HYBRID_RERANKER_ENABLE_CROSS_ENCODER", "true").strip().lower() in {"1", "true", "yes"}
        self.enable_cross_encoder = bool(enable_cross_encoder)

        if self.enable_cross_encoder:
            self._try_load_cross_encoder()

    def _try_load_cross_encoder(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"[HybridReranker] Loading cross-encoder model: {self.cross_encoder_model}")
            self._cross_encoder = CrossEncoder(
                self.cross_encoder_model,
                max_length=self.max_length,
                device="cpu",
            )
            self._cross_encoder_available = True
            print(f"[HybridReranker] Cross-encoder loaded successfully.")
        except Exception as exc:
            print(f"[HybridReranker] Failed to load cross-encoder '{self.cross_encoder_model}': {exc}. Fallback to heuristic only.")
            self._cross_encoder_available = False
            self._cross_encoder = None

    def _build_text(self, context: Dict[str, object]) -> str:
        """Build a single text string for cross-encoder scoring."""
        metadata = dict(context.get("metadata") or {})
        parts = [
            metadata.get("doc_title"),
            metadata.get("citation"),
            metadata.get("article"),
            metadata.get("legal_path"),
            context.get("content"),
        ]
        return "\n".join(str(p or "").strip() for p in parts if str(p or "").strip())

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        if not self._cross_encoder_available or not self._cross_encoder or not candidates:
            return candidates

        texts = [self._build_text(c) for c in candidates]
        pairs = [(query, text) for text in texts]

        started = time.perf_counter()
        raw_scores = self._cross_encoder.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        LOGGER.info("Cross-encoder scored %d candidates in %.2fs", len(candidates), time.perf_counter() - started)

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        # Normalize scores to [0, 1]
        min_score = min(raw_scores) if raw_scores else 0.0
        max_score = max(raw_scores) if raw_scores else 1.0
        score_range = max(1e-8, max_score - min_score)

        for context, raw_score in zip(candidates, raw_scores):
            heuristic_score = float(context.get("final_score") or 0.0)
            normalized_cross = (float(raw_score) - min_score) / score_range
            combined = (
                self.heuristic_weight * heuristic_score
                + self.cross_weight * normalized_cross
            )
            context["cross_encoder_score"] = round(float(raw_score), 4)
            context["normalized_cross_score"] = round(normalized_cross, 4)
            context["heuristic_score"] = round(heuristic_score, 4)
            context["final_score"] = round(combined, 6)

        # Re-sort by combined score
        candidates.sort(key=lambda item: float(item["final_score"]), reverse=True)
        # Print top 3 for visibility
        if candidates:
            print(f"[HybridReranker] Top candidates: {', '.join(candidates[i].get('chunk_id', '?') + f':{candidates[i].get('final_score', 0):.3f}' for i in range(min(3, len(candidates))))}")
        return candidates

    def rerank(
        self,
        query: str,
        contexts: Sequence[Dict[str, object]],
        *,
        max_contexts: int | None = None,
    ) -> List[Dict[str, object]]:
        """Two-stage rerank: heuristic filter then cross-encoder rerank."""
        if not contexts:
            return []

        max_out = max_contexts if max_contexts is not None else self.max_contexts

        # Stage 1: Heuristic reranker (fast, filters to top-K)
        t0 = time.perf_counter()
        heuristic_results = self.heuristic.rerank(
            query,
            contexts,
            max_contexts=self.heuristic_top_k,
        )
        heuristic_time = time.perf_counter() - t0
        print(f"[HybridReranker] Heuristic: {len(contexts)} → {len(heuristic_results)} candidates in {heuristic_time:.3f}s")

        if not heuristic_results:
            return []

        # Stage 2: Cross-encoder rerank (accurate, on top-K only)
        if self.enable_cross_encoder and self._cross_encoder_available:
            try:
                t0 = time.perf_counter()
                cross_results = self._cross_encoder_rerank(query, heuristic_results)
                cross_time = time.perf_counter() - t0
                print(f"[HybridReranker] Cross-encoder: reranked {len(cross_results)} candidates in {cross_time:.3f}s")
                return cross_results[:max_out]
            except Exception as exc:
                print(f"[HybridReranker] Cross-encoder rerank failed: {exc}. Fallback to heuristic.")
                return heuristic_results[:max_out]

        print(f"[HybridReranker] Cross-encoder disabled/unavailable. Using heuristic only.")
        return heuristic_results[:max_out]


def _cli() -> None:
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Hybrid reranker (heuristic + cross-encoder).")
    parser.add_argument("--query", required=True)
    parser.add_argument("--heuristic-top-k", type=int, default=50)
    parser.add_argument("--max-contexts", type=int, default=5)
    args = parser.parse_args()

    demo = [
        {
            "chunk_id": "a",
            "content": "Điều 17. Người không được thành lập doanh nghiệp...",
            "retrieval_score": 0.7,
            "context_type": "seed",
            "metadata": {
                "source_url": "x",
                "citation": "Luật Doanh nghiệp, Điều 17",
                "doc_title": "Luật Doanh nghiệp",
                "domain": "business_law",
            },
        },
        {
            "chunk_id": "b",
            "content": "Tin liên quan doanh nghiệp...",
            "retrieval_score": 0.6,
            "context_type": "neighbor",
            "metadata": {
                "doc_title": "Tin tức",
                "domain": "news",
            },
        },
    ]
    hybrid = HybridReranker(heuristic_top_k=args.heuristic_top_k, max_contexts=args.max_contexts)
    results = hybrid.rerank(args.query, demo)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
