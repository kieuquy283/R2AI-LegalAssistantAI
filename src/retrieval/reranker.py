from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence

from rag.modules.retrieval.utils import tokenize_for_bm25


RELATION_BOOSTS = {
    "seed": 0.20,
    "parent": 0.15,
    "explicit_reference": 0.18,
    "same_article": 0.12,
    "cross_domain": 0.10,
    "neighbor": 0.05,
}


class Reranker:
    def keyword_overlap_score(self, query: str, text: str) -> float:
        query_tokens = set(tokenize_for_bm25(query))
        text_tokens = set(tokenize_for_bm25(text))
        if not query_tokens or not text_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)

    def rerank(self, query: str, contexts: Sequence[Dict[str, object]], *, max_contexts: int = 5) -> List[Dict[str, object]]:
        deduped: Dict[str, Dict[str, object]] = {}
        for context in contexts:
            deduped[str(context["chunk_id"])] = dict(context)

        ranked: List[Dict[str, object]] = []
        for context in deduped.values():
            retrieval_score = float(context.get("retrieval_score") or context.get("score") or 0.0)
            overlap = self.keyword_overlap_score(query, str(context.get("content") or ""))
            relation_boost = RELATION_BOOSTS.get(str(context.get("context_type")), 0.0)
            metadata = dict(context.get("metadata") or {})
            if metadata.get("citation"):
                relation_boost += 0.02
            if metadata.get("source_url"):
                relation_boost += 0.02
            final_score = retrieval_score * 0.55 + overlap * 0.25 + relation_boost * 0.20
            context["rerank_score"] = overlap
            context["relation_boost"] = relation_boost
            context["final_score"] = final_score
            ranked.append(context)

        ranked.sort(key=lambda item: float(item["final_score"]), reverse=True)
        return ranked[:max_contexts]


def _cli() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Heuristic reranker for expanded legal contexts.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    demo = [
        {
            "chunk_id": "a",
            "content": "Điều 17. Người không được thành lập doanh nghiệp...",
            "retrieval_score": 0.7,
            "context_type": "seed",
            "metadata": {"source_url": "x", "citation": "Luật Doanh nghiệp, Điều 17"},
        },
        {
            "chunk_id": "b",
            "content": "Tin liên quan...",
            "retrieval_score": 0.6,
            "context_type": "neighbor",
            "metadata": {},
        },
    ]
    print(json.dumps(Reranker().rerank(args.query, demo), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
