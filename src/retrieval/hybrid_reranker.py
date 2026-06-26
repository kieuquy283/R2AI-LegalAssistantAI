from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Sequence

import requests

from src.retrieval.reranker import Reranker as HeuristicReranker

# Ensure cache directories point to D: drive to avoid C: disk full
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\huggingface_cache\hub")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\huggingface_cache\transformers")
os.environ.setdefault("TORCH_HOME", r"D:\huggingface_cache\torch")

LOGGER = logging.getLogger(__name__)

# Circuit breaker state for API reranker
_API_CIRCUIT_BREAKER = {
    "failures": 0,
    "first_failure_time": 0.0,
    "open_until": 0.0,  # timestamp until which circuit is open
}
_CIRCUIT_BREAKER_THRESHOLD = 3       # consecutive failures to open circuit
_CIRCUIT_BREAKER_WINDOW = 60.0       # seconds to count failures
_CIRCUIT_BREAKER_COOLDOWN = 300.0    # seconds before retrying (5 min)


class HybridReranker:
    """Cascade reranker: heuristic first (fast) then cross-encoder (accurate)."""

    def __init__(
        self,
        *,
        heuristic_top_k: int = 15,
        max_contexts: int = 5,
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 16,
        max_length: int = 256,
        heuristic_weight: float = 0.8,
        cross_weight: float = 0.2,
        enable_cross_encoder: bool | None = None,
        device_strategy: str = "auto",
    ) -> None:
        self.heuristic = HeuristicReranker()
        self.heuristic_top_k = int(os.getenv("HYBRID_RERANKER_HEURISTIC_TOP_K", heuristic_top_k))
        self._skip_heuristic = os.getenv("HYBRID_RERANKER_SKIP_HEURISTIC", "false").strip().lower() in {"1", "true", "yes"}
        self.max_contexts = int(max_contexts)
        self.cross_encoder_model = str(os.getenv("HYBRID_RERANKER_MODEL", cross_encoder_model))
        self.batch_size = int(os.getenv("HYBRID_RERANKER_BATCH_SIZE", batch_size))
        self.max_length = int(os.getenv("HYBRID_RERANKER_MAX_LENGTH", max_length))
        self.heuristic_weight = float(heuristic_weight)
        self.cross_weight = float(os.getenv("HYBRID_RERANKER_CROSS_WEIGHT", cross_weight))
        self._cross_encoder: Any | None = None
        self._cross_encoder_available = False
        self._current_device = "cpu"
        self._device_strategy = str(os.getenv("HYBRID_RERANKER_DEVICE_STRATEGY", device_strategy)).strip().lower()
        
        # API reranker config (SiliconFlow)
        self._api_reranker_enabled = os.getenv("HYBRID_RERANKER_API_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
        self._api_reranker_url = os.getenv("HYBRID_RERANKER_API_URL", "https://api.siliconflow.com/v1/rerank")
        self._api_reranker_key = os.getenv("HYBRID_RERANKER_API_KEY", "")
        self._api_reranker_model = os.getenv("HYBRID_RERANKER_API_MODEL", "Qwen/Qwen3-Reranker-0.6B")

        if enable_cross_encoder is None:
            enable_cross_encoder = os.getenv("HYBRID_RERANKER_ENABLE_CROSS_ENCODER", "true").strip().lower() in {"1", "true", "yes"}
        self.enable_cross_encoder = bool(enable_cross_encoder)

        if self.enable_cross_encoder and not self._api_reranker_enabled:
            self._try_load_cross_encoder()

    def _try_load_cross_encoder(self) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        devices_to_try = []
        if self._device_strategy == "gpu":
            devices_to_try = ["cuda"]
        elif self._device_strategy == "cpu":
            devices_to_try = ["cpu"]
        else:
            # auto: try GPU first, then CPU
            devices_to_try = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

        for device in devices_to_try:
            try:
                print(f"[HybridReranker] Loading cross-encoder model: {self.cross_encoder_model} on {device}")
                self._cross_encoder = CrossEncoder(
                    self.cross_encoder_model,
                    max_length=self.max_length,
                    device=device,
                )
                self._cross_encoder_available = True
                self._current_device = device
                print(f"[HybridReranker] Cross-encoder loaded successfully on {device}.")
                return
            except Exception as exc:
                print(f"[HybridReranker] Failed to load cross-encoder on {device}: {exc}")
                if device == "cuda":
                    print(f"[HybridReranker] GPU load failed. Will try CPU fallback...")
                continue

        print(f"[HybridReranker] All devices failed. Fallback to heuristic only.")
        self._cross_encoder_available = False
        self._cross_encoder = None
        self._current_device = "cpu"

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

    def _fallback_to_cpu(self) -> bool:
        """Try to reload cross-encoder on CPU after GPU failure."""
        if self._current_device == "cpu" or not self._cross_encoder:
            return False
        try:
            import torch
            from sentence_transformers import CrossEncoder
            print(f"[HybridReranker] GPU OOM. Switching to CPU permanently...")
            # Clear GPU memory
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            self._cross_encoder = CrossEncoder(
                self.cross_encoder_model,
                max_length=self.max_length,
                device="cpu",
            )
            self._current_device = "cpu"
            print(f"[HybridReranker] Cross-encoder reloaded successfully on CPU.")
            return True
        except Exception as exc:
            print(f"[HybridReranker] CPU fallback also failed: {exc}")
            self._cross_encoder_available = False
            self._cross_encoder = None
            return False

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        if not self._cross_encoder_available or not self._cross_encoder or not candidates:
            return candidates

        texts = [self._build_text(c) for c in candidates]
        pairs = [(query, text) for text in texts]

        # Try inference with automatic OOM handling
        raw_scores = None
        current_batch_size = self.batch_size
        max_retries = 3

        for attempt in range(max_retries):
            try:
                started = time.perf_counter()
                raw_scores = self._cross_encoder.predict(
                    pairs,
                    batch_size=current_batch_size,
                    show_progress_bar=False,
                )
                LOGGER.info("Cross-encoder scored %d candidates in %.2fs", len(candidates), time.perf_counter() - started)
                break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() or "CUDA" in str(exc):
                    print(f"[HybridReranker] OOM on {self._current_device} (batch={current_batch_size}).")
                    if self._current_device == "cuda":
                        # Try reducing batch size first
                        if current_batch_size > 1:
                            current_batch_size = max(1, current_batch_size // 2)
                            print(f"[HybridReranker] Retrying with batch_size={current_batch_size}...")
                            continue
                        else:
                            # batch_size=1 still OOM on GPU → fallback to CPU
                            if self._fallback_to_cpu():
                                current_batch_size = self.batch_size  # reset batch size for CPU
                                continue
                    else:
                        # CPU OOM (unlikely) → halve batch
                        if current_batch_size > 1:
                            current_batch_size = max(1, current_batch_size // 2)
                            continue
                    print(f"[HybridReranker] Cannot recover from OOM. Using heuristic fallback.")
                    return candidates
                else:
                    raise

        if raw_scores is None:
            return candidates

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        # Normalize scores to [0, 1]
        min_score = min(raw_scores) if raw_scores else 0.0
        max_score = max(raw_scores) if raw_scores else 1.0
        score_range = max(1e-8, max_score - min_score)

        for context, raw_score in zip(candidates, raw_scores):
            heuristic_score = float(context.get("final_score") or 0.0)
            normalized_cross = (float(raw_score) - min_score) / score_range
            context["cross_encoder_score"] = round(float(raw_score), 4)
            context["normalized_cross_score"] = round(normalized_cross, 4)
            context["heuristic_score"] = round(heuristic_score, 4)
            context["final_score"] = round(normalized_cross, 6)

        # Re-sort by cross-encoder score only (cascaded)
        candidates.sort(key=lambda item: float(item["final_score"]), reverse=True)
        # Print top 3 for visibility
        if candidates:
            print(f"[HybridReranker] Top candidates: {', '.join(candidates[i].get('chunk_id', '?') + f':{candidates[i].get('final_score', 0):.3f}' for i in range(min(3, len(candidates))))}")
        return candidates

    def _api_rerank(
        self,
        query: str,
        candidates: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        global _API_CIRCUIT_BREAKER
        now = time.perf_counter()

        # Circuit breaker: check if open
        if _API_CIRCUIT_BREAKER["open_until"] > now:
            print(f"[HybridReranker] API circuit breaker open (until {_API_CIRCUIT_BREAKER['open_until']-now:.0f}s). Fallback to heuristic.")
            return candidates

        if not self._api_reranker_key or not candidates:
            return candidates

        texts = [self._build_text(c) for c in candidates]

        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 1.0
        last_error = None

        result = None
        api_weight = float(os.getenv("R2AI_API_WEIGHT", "0.5"))
        heuristic_weight = 1.0 - api_weight
        t0 = time.perf_counter()
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self._api_reranker_key}",
                    "Content-Type": "application/json",
                }

                # Truncate texts to reduce payload size
                max_text_length = int(os.getenv("R2AI_API_TRUNCATION", "2048"))
                truncated_texts = [t[:max_text_length] for t in texts]

                payload = {
                    "model": self._api_reranker_model,
                    "query": query[:200],
                    "documents": truncated_texts,
                }

                response = requests.post(
                    self._api_reranker_url,
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                # Success: check if we should close circuit
                _API_CIRCUIT_BREAKER["failures"] = 0
                _API_CIRCUIT_BREAKER["first_failure_time"] = 0.0
                break

            except requests.exceptions.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response else 0
                if status_code == 400:
                    print(f"[HybridReranker] API 400 Bad Request (attempt {attempt + 1}/{max_retries}).")
                elif status_code == 429:
                    print(f"[HybridReranker] API rate limited (attempt {attempt + 1}/{max_retries}).")
                else:
                    print(f"[HybridReranker] API error {status_code} (attempt {attempt + 1}/{max_retries}).")

                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[HybridReranker] Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    self._record_api_failure(now)
                    return candidates
            except Exception as exc:
                last_error = exc
                print(f"[HybridReranker] API exception (attempt {attempt + 1}/{max_retries}): {exc}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    self._record_api_failure(now)
                    return candidates

        if result is None:
            self._record_api_failure(now)
            return candidates

        api_time = time.perf_counter() - t0

        # Extract scores from API response (SiliconFlow format)
        results = None
        if "results" in result:
            results = result["results"]
        elif "output" in result and isinstance(result["output"], dict) and "results" in result["output"]:
            results = result["output"]["results"]
        elif "data" in result:
            results = result["data"]

        if not results:
            print(f"[HybridReranker] API response format unexpected: {list(result.keys())}")
            self._record_api_failure(now)
            return candidates

        # Map API results back to candidates
        api_scores = {}
        for r in results:
            idx = r.get("index")
            if idx is not None:
                api_scores[idx] = r.get("relevance_score", r.get("score", 0.0))

        # Blend: configurable weight between API and heuristic
        for i, context in enumerate(candidates):
            heuristic_score = float(context.get("final_score") or 0.0)
            api_score = api_scores.get(i, 0.0)
            context["api_score"] = round(api_score, 4)
            context["heuristic_score"] = round(heuristic_score, 4)
            context["final_score"] = round(api_weight * api_score + heuristic_weight * heuristic_score, 6)

        candidates.sort(key=lambda item: float(item["final_score"]), reverse=True)

        # Dynamic filter: retain candidates with scores above relative threshold
        if candidates:
            best_score = float(candidates[0]["final_score"])
            kept: List[Dict[str, object]] = []
            for c in candidates:
                s = float(c["final_score"])
                if s >= best_score * 0.30:
                    kept.append(c)
            if not kept:
                kept = candidates[:1]
            candidates = kept[:12]

        print(f"[HybridReranker] API rerank: dynamic filter -> {len(candidates)} candidates in {api_time:.3f}s")
        if candidates:
            print(f"[HybridReranker] Top candidates: {', '.join(candidates[i].get('chunk_id', '?') + f':{candidates[i].get('final_score', 0):.3f}' for i in range(min(3, len(candidates))))}")
        return candidates

    def _record_api_failure(self, now: float) -> None:
        """Record API failure and potentially open circuit breaker."""
        global _API_CIRCUIT_BREAKER
        if _API_CIRCUIT_BREAKER["first_failure_time"] == 0.0:
            _API_CIRCUIT_BREAKER["first_failure_time"] = now
        elapsed = now - _API_CIRCUIT_BREAKER["first_failure_time"]

        if elapsed > _CIRCUIT_BREAKER_WINDOW:
            # Window expired, reset
            _API_CIRCUIT_BREAKER["failures"] = 1
            _API_CIRCUIT_BREAKER["first_failure_time"] = now
        else:
            _API_CIRCUIT_BREAKER["failures"] += 1

        if _API_CIRCUIT_BREAKER["failures"] >= _CIRCUIT_BREAKER_THRESHOLD:
            _API_CIRCUIT_BREAKER["open_until"] = now + _CIRCUIT_BREAKER_COOLDOWN
            print(f"[HybridReranker] Circuit breaker OPEN ({_CIRCUIT_BREAKER_COOLDOWN:.0f}s cooldown).")
        print(f"[HybridReranker] API failures: {_API_CIRCUIT_BREAKER['failures']}/{_CIRCUIT_BREAKER_THRESHOLD}")

    # Difficulty-to-heuristic_top_k map for adaptive recall
    _DIFFICULTY_TOPK_MAP = {
        "easy": 10,
        "mid": 25,
        "hard": 60,
        "very_hard": 120,
    }

    def _get_effective_top_k(self, difficulty: str | None) -> int:
        if difficulty and difficulty in self._DIFFICULTY_TOPK_MAP:
            return self._DIFFICULTY_TOPK_MAP[difficulty]
        return self.heuristic_top_k

    def _consensus_pre_filter(
        self,
        contexts: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        result = []
        for c in contexts:
            dense = float(c.get("dense_score") or 0.0)
            sparse = float(c.get("bm25_score") or 0.0)
            exact = float(c.get("exact_score") or 0.0)
            sources = sum(1 for s in [dense, sparse, exact] if s > 0)
            consensus = sources / 3.0
            c["consensus_score"] = round(consensus, 4)
            raw_fusion = float(c.get("final_score") or 0.0)
            c["raw_fusion_score"] = round(raw_fusion, 6)
            c["final_score"] = round(0.6 * raw_fusion + 0.4 * consensus, 6)
            result.append(c)
        result.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
        return result

    def rerank(
        self,
        query: str,
        contexts: Sequence[Dict[str, object]],
        *,
        max_contexts: int | None = None,
        difficulty: str | None = None,
    ) -> List[Dict[str, object]]:
        """Two-stage rerank: consensus pre-filter → heuristic → cross-encoder/API."""
        if not contexts:
            return []

        max_out = max_contexts if max_contexts is not None else self.max_contexts
        effective_top_k = self._get_effective_top_k(difficulty)

        # Consensus pre-filter: boost multi-source matches before heuristic cutoff
        contexts = self._consensus_pre_filter(list(contexts))

        # Skip heuristic entirely if flag is set (send all candidates to API)
        if self._skip_heuristic:
            if self._api_reranker_enabled and self._api_reranker_key:
                try:
                    t0 = time.perf_counter()
                    api_results = self._api_rerank(query, list(contexts))
                    api_time = time.perf_counter() - t0
                    print(f"[HybridReranker] Skipped heuristic. API reranker: {len(api_results)} candidates in {api_time:.3f}s")
                    return api_results[:max_out]
                except Exception as exc:
                    print(f"[HybridReranker] API rerank failed: {exc}. Using heuristic fallback.")
            else:
                print(f"[HybridReranker] Skip heuristic set but no API reranker configured. Using heuristic.")
            # Fall through to normal flow if API unavailable or failed

        # Stage 1: Heuristic reranker (fast, filters to top-K)
        t0 = time.perf_counter()
        heuristic_results = self.heuristic.rerank(
            query,
            contexts,
            max_contexts=effective_top_k,
        )
        heuristic_time = time.perf_counter() - t0
        print(f"[HybridReranker] Heuristic: {len(contexts)} -> {len(heuristic_results)} candidates in {heuristic_time:.3f}s")

        if not heuristic_results:
            return []

        # Stage 2: API or Cross-encoder rerank (accurate, on top-K only)
        if self._api_reranker_enabled and self._api_reranker_key:

            try:
                t0 = time.perf_counter()
                api_results = self._api_rerank(query, heuristic_results)
                api_time = time.perf_counter() - t0
                print(f"[HybridReranker] API reranker: reranked {len(api_results)} candidates in {api_time:.3f}s")
                return api_results[:max_out]
            except Exception as exc:
                print(f"[HybridReranker] API rerank failed: {exc}. Fallback to heuristic.")
                return heuristic_results[:max_out]
        
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

        print(f"[HybridReranker] Cross-encoder/API disabled/unavailable. Using heuristic only.")
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
