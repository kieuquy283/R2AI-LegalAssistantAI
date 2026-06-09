from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from src.retrieval.confidence_checker import ConfidenceChecker
from src.retrieval.context_expander import ContextExpander
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_router import route_query
from src.retrieval.reranker import Reranker


class RetrievalPipeline:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.expander = ContextExpander(retriever=self.retriever)
        self.reranker = Reranker()
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
        if not final_contexts:
            fallback_contexts = list(expanded_contexts or seed_chunks)
            final_contexts = fallback_contexts[:max_contexts]
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
