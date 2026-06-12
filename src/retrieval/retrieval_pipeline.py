from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from rag.config.runtime import get_retrieval_runtime_config
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.confidence_checker import ConfidenceChecker
from src.retrieval.context_expander import ContextExpander
from src.retrieval.hybrid_fusion import fuse_candidates, select_dynamic_contexts
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.legal_exact_search import LegalExactSearch
from src.retrieval.qdrant_retriever import QdrantRetriever, apply_domain_adjustment
from src.retrieval.query_router import route_query
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_reranker import HybridReranker


class RetrievalPipeline:
    def __init__(self) -> None:
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
        else:
            self.retriever = HybridRetriever()
            self.expander = ContextExpander(retriever=self.retriever)
            self.reranker = HybridReranker() if self._use_hybrid_reranker else Reranker()
            self.confidence_checker = ConfidenceChecker()

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

    def _run_qdrant(self, query: str) -> Dict[str, object]:
        import time
        t_total = time.perf_counter()
        
        t0 = time.perf_counter()
        initial_route = route_query(query, seed_chunks=[])
        preferred_domains = list(initial_route.get("domains") or [])
        t_route = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        dense_candidates = self.qdrant_retriever.search(query, preferred_domains=preferred_domains) if self.qdrant_retriever else []
        t_dense = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        sparse_candidates = (
            self.bm25_retriever.search(query, top_k=self.runtime_config.candidate_k_sparse, preferred_domains=preferred_domains)
            if self.bm25_retriever
            else []
        )
        t_sparse = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        # Skip exact search if disabled or query has no legal ref pattern
        skip_exact = self.runtime_config.candidate_k_title <= 0
        if not skip_exact and self.exact_search:
            from src.retrieval.legal_exact_search import LEGAL_REF_PATTERN
            skip_exact = not LEGAL_REF_PATTERN.search(query)
        exact_candidates = (
            self.exact_search.search(query, top_k=self.runtime_config.candidate_k_title, preferred_domains=preferred_domains)
            if self.exact_search and not skip_exact
            else []
        )
        t_exact = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        apply_domain_adjustment(query, dense_candidates)
        t_domain = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        reranked = fuse_candidates(
            query,
            dense_candidates=dense_candidates,
            bm25_candidates=sparse_candidates,
            exact_candidates=exact_candidates,
            preferred_domains=preferred_domains,
            config=self.runtime_config,
        )
        t_fuse = time.perf_counter() - t0
        
        # Optional hybrid reranker pass on fused candidates
        if self._use_hybrid_reranker:
            t0 = time.perf_counter()
            if not self.reranker:
                self.reranker = HybridReranker()
            reranked = self.reranker.rerank(query, reranked, max_contexts=self.runtime_config.candidate_k_chunks)
            t_rerank = time.perf_counter() - t0
        else:
            t_rerank = 0.0
        
        t0 = time.perf_counter()
        final_contexts = select_dynamic_contexts(reranked, config=self.runtime_config)
        t_select = time.perf_counter() - t0
        
        t_total = time.perf_counter() - t_total
        print(f"[Retrieval] route={t_route:.3f}s dense={t_dense:.3f}s sparse={t_sparse:.3f}s exact={t_exact:.3f}s domain={t_domain:.3f}s fuse={t_fuse:.3f}s rerank={t_rerank:.3f}s select={t_select:.3f}s TOTAL={t_total:.3f}s")
        
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
            "dense_candidates": dense_candidates,
            "sparse_candidates": sparse_candidates,
            "exact_candidates": exact_candidates,
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
