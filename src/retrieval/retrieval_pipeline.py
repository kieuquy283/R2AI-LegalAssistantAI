from __future__ import annotations

import argparse
import json
from typing import Dict

from src.retrieval.context_expander import ContextExpander
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_router import route_query
from src.retrieval.reranker import Reranker


class RetrievalPipeline:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.expander = ContextExpander(retriever=self.retriever)
        self.reranker = Reranker()

    def run(self, query: str) -> Dict[str, object]:
        seed_chunks = self.retriever.search(query, top_k=5)
        route = route_query(query, seed_chunks=seed_chunks)
        expanded_contexts = self.expander.expand(query=query, route_result=route, seed_chunks=seed_chunks)
        max_contexts = {
            "SIMPLE_VECTOR": 5,
            "PARENT_CONTEXT": 7,
            "LEGAL_GRAPH_CONTEXT": 9,
            "CROSS_DOMAIN_CONTEXT": 10,
            "MULTI_DOMAIN_COMPLEX": 12,
        }.get(route["route"], 7)
        final_contexts = self.reranker.rerank(query, expanded_contexts, max_contexts=max_contexts)
        return {
            "query": query,
            "route": route["route"],
            "domains": route["domains"],
            "route_result": route,
            "seed_chunks": seed_chunks,
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
