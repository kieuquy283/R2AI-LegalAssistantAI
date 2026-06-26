from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from rag.config.runtime import get_retrieval_runtime_config
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.confidence_checker import ConfidenceChecker
from src.retrieval.context_expander import ContextExpander
from src.retrieval.hybrid_fusion import fuse_candidates, select_dynamic_contexts
from src.retrieval.hybrid_fusion import _estimate_difficulty
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.legal_exact_search import LegalExactSearch
from src.retrieval.qdrant_retriever import QdrantRetriever, apply_domain_adjustment
from src.retrieval.query_classifier import classify_query
from src.retrieval.query_expander import expand_query
from src.retrieval.query_router import route_query
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_reranker import HybridReranker
from src.generation.llm_client import LLMClient


_RETRIEVAL_PIPELINE_INSTANCE: RetrievalPipeline | None = None

# Config flags
_USE_MULTI_QUERY = os.getenv("R2AI_USE_MULTI_QUERY", "true").strip().lower() in {"1", "true", "yes"}
_USE_CRAG = os.getenv("R2AI_USE_CRAG", "true").strip().lower() in {"1", "true", "yes"}
_USE_PARALLEL_RETRIEVAL = os.getenv("R2AI_USE_PARALLEL_RETRIEVAL", "true").strip().lower() in {"1", "true", "yes"}
_CRAG_MIN_CONTEXTS = int(os.getenv("R2AI_CRAG_MIN_CONTEXTS", "2"))


def _generate_multi_queries(query: str, difficulty: str) -> List[str]:
    """Task 4: Generate multiple query variants for better recall."""
    if not _USE_MULTI_QUERY or difficulty in ("easy",):
        return [query]
    client = LLMClient(temperature=0.3)
    if not client.is_available():
        return [query]
    system_prompt = "Bạn là chuyên gia pháp lý Việt Nam. Chỉ trả về danh sách JSON, không thêm giải thích."
    user_prompt = (
        f"Câu hỏi gốc: {query}\n\n"
        "Hãy tạo 3 cách diễn đạt khác nhau của cùng câu hỏi này để tìm kiếm văn bản pháp luật hiệu quả hơn.\n"
        "Mỗi biến thể nên:\n"
        "- Dùng từ đồng nghĩa pháp lý\n"
        "- Thêm/bớt từ khóa mà vẫn giữ nguyên ý định\n"
        "- Làm rõ số hiệu văn bản nếu có\n"
        "Trả về JSON array: [\"biến thể 1\", \"biến thể 2\", \"biến thể 3\"]"
    )
    try:
        text = client.generate(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.3)
        if text:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            variants = json.loads(text)
            if isinstance(variants, list) and len(variants) >= 1:
                queries = [query] + [str(v).strip() for v in variants if str(v).strip()]
                print(f"[MultiQuery] Generated {len(queries)} variants for difficulty={difficulty}")
                return queries
    except Exception as exc:
        print(f"[MultiQuery] Generation failed: {exc}")
    return [query]


def _crag_refine_query(initial_query: str, initial_contexts: List[Dict]) -> str | None:
    """Task 5: If too few contexts, refine query for re-retrieval."""
    if not _USE_CRAG or len(initial_contexts) >= _CRAG_MIN_CONTEXTS:
        return None
    client = LLMClient(temperature=0.0)
    if not client.is_available():
        return None
    system_prompt = "Bạn là chuyên gia pháp lý. Chỉ trả về câu đã viết lại."
    user_prompt = (
        f"Câu hỏi gốc: {initial_query}\n\n"
        f"Số lượng kết quả tìm được: {len(initial_contexts)} (quá ít).\n"
        "Hãy viết lại câu hỏi để mở rộng phạm vi tìm kiếm:\n"
        "- Thêm từ khóa đồng nghĩa\n"
        "- Giảm specificity (bỏ bớt chi tiết thừa)\n"
        "- Dùng thuật ngữ pháp lý rộng hơn\n\n"
        "Câu hỏi mở rộng:"
    )
    try:
        refined = client.generate(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
        if refined and len(refined) > 10:
            print(f"[CRAG] Refined query: '{initial_query[:80]}...' -> '{refined[:120]}...'")
            return refined.strip()
    except Exception as exc:
        print(f"[CRAG] Refinement failed: {exc}")
    return None


class RetrievalPipeline:
    def __new__(cls) -> RetrievalPipeline:
        global _RETRIEVAL_PIPELINE_INSTANCE
        if _RETRIEVAL_PIPELINE_INSTANCE is not None:
            return _RETRIEVAL_PIPELINE_INSTANCE
        instance = super().__new__(cls)
        _RETRIEVAL_PIPELINE_INSTANCE = instance
        return instance

    def __init__(self) -> None:
        # Prevent re-initialization if singleton already initialized
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.runtime_config = get_retrieval_runtime_config()
        self.backend = str(self.runtime_config.retrieval_backend or "faiss").strip().lower()
        self.retriever = None
        self.expander = None
        self.reranker = None
        self.confidence_checker = None
        self.qdrant_retriever = None
        self.bm25_retriever = None
        self.exact_search = None
        self._use_hybrid_reranker = os.getenv("HYBRID_RERANKER", "true").strip().lower() not in {"0", "false", "no", "off"}
        if self.backend == "qdrant":
            self.qdrant_retriever = QdrantRetriever(config=self.runtime_config)
            self.bm25_retriever = BM25Retriever()
            self.exact_search = LegalExactSearch()
            if self._use_hybrid_reranker:
                self.reranker = HybridReranker()
            # Pre-load BM25 index at startup to avoid first-query latency
            try:
                print("[RetrievalPipeline] Pre-loading BM25 index...")
                self.bm25_retriever.preload()
            except Exception as exc:
                print(f"[RetrievalPipeline] BM25 pre-load failed: {exc}. Will load on first query.")
        else:
            self.retriever = HybridRetriever()
            self.expander = ContextExpander(retriever=self.retriever)
            self.reranker = HybridReranker() if self._use_hybrid_reranker else Reranker()
            self.confidence_checker = ConfidenceChecker()

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        global _RETRIEVAL_PIPELINE_INSTANCE
        _RETRIEVAL_PIPELINE_INSTANCE = None

    def _route_result_for(self, route: str, domains: list[str], reason: str) -> Dict[str, object]:
        needs_parent = route in {"PARENT_CONTEXT", "LEGAL_GRAPH_CONTEXT", "CROSS_DOMAIN_CONTEXT", "MULTI_DOMAIN_COMPLEX"}
        needs_neighbor = route == "MULTI_DOMAIN_COMPLEX"
        needs_graph = route in {"LEGAL_GRAPH_CONTEXT", "CROSS_DOMAIN_CONTEXT", "MULTI_DOMAIN_COMPLEX"}
        needs_cross_domain = route in {"CROSS_DOMAIN_CONTEXT", "MULTI_DOMAIN_COMPLEX"}
        return {
            "route": route,
            "domains": domains,
            "needs_parent": needs_parent,
            "needs_neighbor": needs_neighbor,
            "needs_graph": needs_graph,
            "needs_cross_domain": needs_cross_domain,
            "reason": reason,
        }

    def run(self, query: str) -> Dict[str, object]:
        if self.backend == "qdrant":
            return self._run_qdrant(query)
        seed_top_k = max(1, int(os.getenv("R2AI_RETRIEVAL_TOP_K", "5")))
        max_contexts_override = os.getenv("R2AI_RETRIEVAL_MAX_CONTEXTS")
        skip_expansion = os.getenv("R2AI_RETRIEVAL_SKIP_EXPANSION", "").strip().lower() in {"1", "true", "yes"}
        skip_rerank = os.getenv("R2AI_RETRIEVAL_SKIP_RERANKING", "").strip().lower() in {"1", "true", "yes"}
        seed_chunks = self.retriever.search(query, top_k=seed_top_k)
        initial_route = route_query(query, seed_chunks=seed_chunks)
        confidence_result = self.confidence_checker.check(
            query=query,
            route_result=initial_route,
            seed_chunks=seed_chunks,
        )
        route = initial_route
        if confidence_result["should_escalate"]:
            route = self._route_result_for(
                str(confidence_result["recommended_route"]),
                list(initial_route.get("domains") or []),
                "Escalated after low-confidence initial retrieval.",
            )
        expanded_contexts = list(seed_chunks) if skip_expansion else self.expander.expand(query=query, route_result=route, seed_chunks=seed_chunks)
        max_contexts = {
            "SIMPLE_VECTOR": 5,
            "PARENT_CONTEXT": 7,
            "LEGAL_GRAPH_CONTEXT": 9,
            "CROSS_DOMAIN_CONTEXT": 10,
            "MULTI_DOMAIN_COMPLEX": 12,
        }.get(route["route"], 7)
        if max_contexts_override:
            try:
                max_contexts = max(1, int(max_contexts_override))
            except ValueError:
                pass
        final_contexts = list(expanded_contexts) if skip_rerank else self.reranker.rerank(query, expanded_contexts, max_contexts=max_contexts)
        return {
            "query": query,
            "route": route["route"],
            "domains": route["domains"],
            "initial_route_result": initial_route,
            "route_result": route,
            "confidence_result": confidence_result,
            "seed_chunks": seed_chunks,
            "seed_contexts": seed_chunks,
            "expanded_contexts": expanded_contexts,
            "final_contexts": final_contexts,
        }

    def _retrieve_single(self, q: str, preferred_domains: list[str], difficulty: str = "mid") -> Dict[str, List]:
        """Run dense + sparse + exact for a single query variant."""
        dense = self.qdrant_retriever.search(q, preferred_domains=preferred_domains, difficulty=difficulty) if self.qdrant_retriever else []
        sparse = (
            self.bm25_retriever.search(q, top_k=self.runtime_config.candidate_k_sparse, preferred_domains=preferred_domains)
            if self.bm25_retriever
            else []
        )
        skip_exact = self.runtime_config.candidate_k_title <= 0
        if not skip_exact and self.exact_search:
            from src.retrieval.legal_exact_search import LEGAL_REF_PATTERN
            skip_exact = not LEGAL_REF_PATTERN.search(q)
        exact = (
            self.exact_search.search(q, top_k=self.runtime_config.candidate_k_title, preferred_domains=preferred_domains)
            if self.exact_search and not skip_exact
            else []
        )
        return {"dense": dense, "sparse": sparse, "exact": exact}

    def _run_qdrant(self, query: str) -> Dict[str, object]:
        t_total = time.perf_counter()

        # Optional: force SIMPLE_VECTOR route (disable adaptive routing)
        force_simple = os.getenv("R2AI_FORCE_SIMPLE_ROUTE", "").strip().lower() in {"1", "true", "yes"}
        if force_simple:
            initial_route = {
                "route": "SIMPLE_VECTOR",
                "domains": [],
                "needs_parent": False,
                "needs_neighbor": False,
                "needs_graph": False,
                "needs_cross_domain": False,
                "reason": "Forced SIMPLE_VECTOR by R2AI_FORCE_SIMPLE_ROUTE",
            }
            print("[Retrieval] Adaptive routing disabled, forced SIMPLE_VECTOR")
        else:
            initial_route = route_query(query, seed_chunks=[])
        preferred_domains = list(initial_route.get("domains") or [])
        t_route = time.perf_counter() - t_total

        # Adaptive depth based on difficulty (Task 6)
        difficulty = _estimate_difficulty(initial_route.get("route"), query)
        depth_scale = {"easy": 0.5, "mid": 1.0, "hard": 1.5, "very_hard": 2.0}.get(difficulty, 1.0)
        adapted_rerank_n = max(50, int(self.runtime_config.rerank_top_n * depth_scale))
        print(f"[Retrieval] Difficulty={difficulty}, adapted_rerank_n={adapted_rerank_n}")

        # Dynamic BM25 bigrams: enable for mid/hard to boost recall
        if difficulty in ("mid", "hard", "very_hard"):
            os.environ["R2AI_BM25_BIGRAMS"] = "true"
        else:
            os.environ["R2AI_BM25_BIGRAMS"] = "false"

        # Query classification for adaptive strategy
        query_class = classify_query(query)
        if query_class["is_specific"]:
            print(f"[Retrieval] Query types: {query_class['types']}")

        # Query Expansion
        expanded_query = expand_query(query, difficulty=difficulty)
        if query_class["boost_exact"]:
            # Ensure exact legal ref is preserved in expanded query
            pass

        # Type-specific keyword injection to improve BM25 recall
        type_keywords = {
            "muc_phat": ["mức phạt tiền", "xử phạt vi phạm hành chính", "chế tài"],
            "thu_tuc": ["thủ tục hành chính", "trình tự thực hiện", "hồ sơ"],
            "dinh_nghia": ["quy định", "theo quy định của pháp luật"],
        }
        extra_kws = []
        for t in query_class.get("types", []):
            extra_kws.extend(type_keywords.get(t, []))
        if extra_kws and not expanded_query.endswith(" ".join(extra_kws)):
            expanded_query = f"{expanded_query} {' '.join(extra_kws)}"
            print(f"[Retrieval] Type-enriched query: -> '{expanded_query}'")

        if expanded_query != query:
            print(f"[Retrieval] Query expanded: '{query}' -> '{expanded_query}'")

        t_expand = time.perf_counter()

        # Task 4: Multi-query variants
        queries = _generate_multi_queries(expanded_query, difficulty)
        t_multi = time.perf_counter() - t_expand

        # Task 7: Parallel retrieval across queries and retriever types
        all_dense: List[Dict] = []
        all_sparse: List[Dict] = []
        all_exact: List[Dict] = []

        t_retrieval_start = time.perf_counter()
        if _USE_PARALLEL_RETRIEVAL and len(queries) > 1:
            with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
                fut_to_q = {executor.submit(self._retrieve_single, q, preferred_domains, difficulty): q for q in queries}
                for fut in as_completed(fut_to_q):
                    q = fut_to_q[fut]
                    try:
                        res = fut.result()
                        all_dense.extend(res["dense"])
                        all_sparse.extend(res["sparse"])
                        all_exact.extend(res["exact"])
                    except Exception as exc:
                        print(f"[Retrieval] Parallel query '{q[:60]}' failed: {exc}")
        else:
            for q in queries:
                dense = self.qdrant_retriever.search(q, preferred_domains=preferred_domains, difficulty=difficulty) if self.qdrant_retriever else []
                all_dense.extend(dense)
                sparse = (
                    self.bm25_retriever.search(q, top_k=self.runtime_config.candidate_k_sparse, preferred_domains=preferred_domains)
                    if self.bm25_retriever
                    else []
                )
                all_sparse.extend(sparse)
                skip_exact = self.runtime_config.candidate_k_title <= 0
                if not skip_exact and self.exact_search:
                    from src.retrieval.legal_exact_search import LEGAL_REF_PATTERN
                    skip_exact = not LEGAL_REF_PATTERN.search(q)
                exact = (
                    self.exact_search.search(q, top_k=self.runtime_config.candidate_k_title, preferred_domains=preferred_domains)
                    if self.exact_search and not skip_exact
                    else []
                )
                all_exact.extend(exact)
        t_retrieval = time.perf_counter() - t_retrieval_start

        # Deduplicate within each list
        seen_ids = set()
        deduped_dense = []
        for c in all_dense:
            cid = c.get("candidate_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                deduped_dense.append(c)
            elif not cid:
                deduped_dense.append(c)

        seen_ids_sparse = set()
        deduped_sparse = []
        for c in all_sparse:
            cid = c.get("candidate_id")
            if cid and cid not in seen_ids_sparse:
                seen_ids_sparse.add(cid)
                deduped_sparse.append(c)
            elif not cid:
                deduped_sparse.append(c)

        seen_ids_exact = set()
        deduped_exact = []
        for c in all_exact:
            cid = c.get("candidate_id")
            if cid and cid not in seen_ids_exact:
                seen_ids_exact.add(cid)
                deduped_exact.append(c)
            elif not cid:
                deduped_exact.append(c)

        t_domain_start = time.perf_counter()
        apply_domain_adjustment(query, deduped_dense)
        t_domain = time.perf_counter() - t_domain_start

        t_fuse_start = time.perf_counter()
        reranked = fuse_candidates(
            query,
            dense_candidates=deduped_dense,
            bm25_candidates=deduped_sparse,
            exact_candidates=deduped_exact,
            preferred_domains=preferred_domains,
            config=self.runtime_config,
        )
        t_fuse = time.perf_counter() - t_fuse_start

        # Adaptive limit: cap fused results before reranker
        reranked = reranked[:adapted_rerank_n]

        # Optional hybrid reranker pass on fused candidates
        if self._use_hybrid_reranker:
            rerank_start = time.perf_counter()
            if not self.reranker:
                self.reranker = HybridReranker()
            rerank_max = self.runtime_config.candidate_k_chunks or self.runtime_config.candidate_k_articles or self.runtime_config.rerank_top_n or 50
            reranked = self.reranker.rerank(query, reranked, max_contexts=rerank_max)
            t_rerank = time.perf_counter() - rerank_start
        else:
            t_rerank = 0.0

        t_select_start = time.perf_counter()
        final_contexts = select_dynamic_contexts(
            reranked,
            route=initial_route.get("route"),
            query=query,
            config=self.runtime_config,
        )
        t_select = time.perf_counter() - t_select_start

        # Task 5: CRAG — refine and re-retrieve if too few contexts
        crag_used = False
        refined_query = _crag_refine_query(query, final_contexts)
        if refined_query and refined_query != query:
            print(f"[CRAG] Re-retrieving with refined query: '{refined_query[:100]}...'")
            q2 = expand_query(refined_query, difficulty=difficulty)
            crag_res = self._retrieve_single(q2, preferred_domains, difficulty)
            apply_domain_adjustment(q2, crag_res["dense"])
            crag_reranked = fuse_candidates(
                q2,
                dense_candidates=crag_res["dense"],
                bm25_candidates=crag_res["sparse"],
                exact_candidates=crag_res["exact"],
                preferred_domains=preferred_domains,
                config=self.runtime_config,
            )[:adapted_rerank_n]
            if self._use_hybrid_reranker and self.reranker:
                crag_reranked = self.reranker.rerank(q2, crag_reranked, max_contexts=rerank_max)
            crag_contexts = select_dynamic_contexts(
                crag_reranked,
                route=initial_route.get("route"),
                query=query,
                config=self.runtime_config,
            )
            if len(crag_contexts) > len(final_contexts):
                print(f"[CRAG] Improved: {len(final_contexts)} -> {len(crag_contexts)} contexts")
                final_contexts = crag_contexts
                crag_used = True
                reranked = crag_reranked

        t_total = time.perf_counter() - t_total
        print(f"[Retrieval] route={t_route:.3f}s multi={t_multi:.3f}s retrieve={t_retrieval:.3f}s domain={t_domain:.3f}s fuse={t_fuse:.3f}s rerank={t_rerank:.3f}s select={t_select:.3f}s CRAG={crag_used} TOTAL={t_total:.3f}s")

        confidence_result = {
            "is_confident": bool(final_contexts),
            "should_escalate": False,
            "recommended_route": initial_route.get("route"),
            "score": float(final_contexts[0].get("final_score") or 0.0) if final_contexts else 0.0,
        }
        return {
            "query": query,
            "route": initial_route["route"],
            "domains": initial_route["domains"],
            "initial_route_result": initial_route,
            "route_result": initial_route,
            "confidence_result": confidence_result,
            "seed_chunks": reranked[: min(5, len(reranked))],
            "seed_contexts": reranked[: min(5, len(reranked))],
            "expanded_contexts": reranked,
            "final_contexts": final_contexts,
            "dense_candidates": deduped_dense,
            "sparse_candidates": deduped_sparse,
            "exact_candidates": deduped_exact,
            "crag_used": crag_used,
            "multi_queries": queries if len(queries) > 1 else None,
        }


def _cli() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run retrieval pipeline for a legal question.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    print(json.dumps(RetrievalPipeline().run(args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
