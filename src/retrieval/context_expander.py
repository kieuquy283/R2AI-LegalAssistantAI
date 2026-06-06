from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from rag.modules.retrieval.utils import tokenize_for_bm25
from src.ingestion.common import read_jsonl
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_router import route_query


class ContextExpander:
    def __init__(
        self,
        *,
        chunks_path: str | Path = "data/processed/chunks.jsonl",
        context_chunks_path: str | Path = "data/processed/context_chunks.jsonl",
        edges_path: str | Path = "data/processed/legal_edges.jsonl",
        cross_domain_edges_path: str | Path = "data/processed/cross_domain_edges.jsonl",
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.chunks = read_jsonl(chunks_path)
        self.context_chunks = read_jsonl(context_chunks_path)
        self.edges = read_jsonl(edges_path)
        self.cross_domain_edges = read_jsonl(cross_domain_edges_path) if Path(cross_domain_edges_path).exists() else []
        self.retriever = retriever or HybridRetriever()

        self.chunk_by_id = {str(row["chunk_id"]): row for row in self.chunks}
        self.context_by_id = {str(row["context_chunk_id"]): row for row in self.context_chunks}
        self.cross_domain_map: Dict[str, List[str]] = {}
        for edge in self.cross_domain_edges:
            self.cross_domain_map.setdefault(str(edge["source_id"]), []).append(str(edge["target_domain"]))

    def _budget_for_route(self, route: str) -> int:
        return {
            "SIMPLE_VECTOR": 5,
            "PARENT_CONTEXT": 7,
            "LEGAL_GRAPH_CONTEXT": 9,
            "CROSS_DOMAIN_CONTEXT": 10,
            "MULTI_DOMAIN_COMPLEX": 12,
        }.get(route, 7)

    def _context_payload(self, chunk_id: str, content: str, metadata: Dict[str, object], *, context_type: str, relation_type: str | None, retrieval_score: float = 0.0) -> Dict[str, object]:
        return {
            "chunk_id": chunk_id,
            "content": content,
            "context_type": context_type,
            "relation_type": relation_type,
            "score": retrieval_score,
            "retrieval_score": retrieval_score,
            "metadata": metadata,
        }

    def _seed_payload(self, seed: Dict[str, object]) -> Dict[str, object]:
        return self._context_payload(
            str(seed["chunk_id"]),
            str(seed["content"]),
            dict(seed.get("metadata") or {}),
            context_type="seed",
            relation_type=None,
            retrieval_score=float(seed.get("retrieval_score") or seed.get("score") or 0.0),
        )

    def _chunk_metadata(self, chunk: Dict[str, object]) -> Dict[str, object]:
        return {
            "doc_id": chunk.get("doc_id"),
            "domain": chunk.get("domain"),
            "doc_title": chunk.get("doc_title"),
            "article": chunk.get("article"),
            "clause": chunk.get("clause"),
            "citation": chunk.get("citation"),
            "source_url": chunk.get("source_url"),
        }

    def _lexical_score(self, query: str, text: str) -> float:
        q = set(tokenize_for_bm25(query))
        t = set(tokenize_for_bm25(text))
        if not q or not t:
            return 0.0
        return len(q & t) / len(q)

    def expand(self, *, query: str, route_result: Dict[str, object], seed_chunks: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        budget = self._budget_for_route(str(route_result["route"]))
        results: List[Dict[str, object]] = []
        seen = set()

        def add(item: Dict[str, object]) -> None:
            key = str(item["chunk_id"])
            if key in seen:
                return
            seen.add(key)
            results.append(item)

        for seed in seed_chunks:
            add(self._seed_payload(seed))

        if route_result.get("needs_parent"):
            for seed in seed_chunks:
                chunk = self.chunk_by_id.get(str(seed["chunk_id"]))
                if not chunk:
                    continue
                context_chunk_id = chunk.get("context_chunk_id")
                if context_chunk_id and str(context_chunk_id) in self.context_by_id:
                    ctx = self.context_by_id[str(context_chunk_id)]
                    ctx_content = str(ctx.get("content") or "") or str(chunk.get("content") or "")
                    add(
                        self._context_payload(
                            str(ctx["context_chunk_id"]),
                            ctx_content,
                            {
                                "doc_id": ctx.get("doc_id"),
                                "domain": ctx.get("domain"),
                                "doc_title": ctx.get("doc_title"),
                                "article": ctx.get("article"),
                                "citation": ctx.get("citation"),
                                "source_url": ctx.get("source_url"),
                            },
                            context_type="parent",
                            relation_type="HAS_PARENT",
                            retrieval_score=float(seed.get("score") or 0.0),
                        )
                    )

        if route_result.get("needs_graph"):
            for seed in seed_chunks:
                chunk = self.chunk_by_id.get(str(seed["chunk_id"]))
                if not chunk:
                    continue
                for ref in list(chunk.get("explicit_refs") or [])[:3]:
                    target_chunk_id = ref.get("target_chunk_id")
                    target_chunk = self.chunk_by_id.get(str(target_chunk_id)) if target_chunk_id else None
                    if not target_chunk:
                        continue
                    add(
                        self._context_payload(
                            str(target_chunk["chunk_id"]),
                            str(target_chunk.get("content") or ""),
                            self._chunk_metadata(target_chunk),
                            context_type="explicit_reference",
                            relation_type="REFERS_TO",
                            retrieval_score=float(seed.get("score") or 0.0),
                        )
                    )

        if route_result.get("needs_cross_domain"):
            satellite_domains = [domain for domain in route_result.get("domains", []) if domain != "business_law"]
            candidates: List[Dict[str, object]] = []
            for chunk in self.chunks:
                mapped_domains = set(self.cross_domain_map.get(str(chunk["chunk_id"]), []))
                if not mapped_domains.intersection(satellite_domains):
                    continue
                candidates.append(chunk)
            candidates.sort(key=lambda chunk: self._lexical_score(query, str(chunk.get("embedding_text") or chunk.get("content") or "")), reverse=True)
            per_domain_counts: Dict[str, int] = {}
            for chunk in candidates:
                mapped_domains = [domain for domain in self.cross_domain_map.get(str(chunk["chunk_id"]), []) if domain in satellite_domains]
                if not mapped_domains:
                    continue
                selected_domain = mapped_domains[0]
                if per_domain_counts.get(selected_domain, 0) >= 2:
                    continue
                per_domain_counts[selected_domain] = per_domain_counts.get(selected_domain, 0) + 1
                add(
                    self._context_payload(
                        str(chunk["chunk_id"]),
                        str(chunk.get("content") or ""),
                        self._chunk_metadata(chunk),
                        context_type="cross_domain",
                        relation_type="RELATED_DOMAIN",
                        retrieval_score=self._lexical_score(query, str(chunk.get("content") or "")),
                    )
                )

        if route_result.get("needs_neighbor"):
            for seed in seed_chunks:
                chunk = self.chunk_by_id.get(str(seed["chunk_id"]))
                if not chunk:
                    continue
                for neighbor_key in ["prev_chunk_id", "next_chunk_id"]:
                    neighbor_id = chunk.get(neighbor_key)
                    neighbor = self.chunk_by_id.get(str(neighbor_id)) if neighbor_id else None
                    if not neighbor or neighbor.get("doc_id") != chunk.get("doc_id"):
                        continue
                    add(
                        self._context_payload(
                            str(neighbor["chunk_id"]),
                            str(neighbor.get("content") or ""),
                            self._chunk_metadata(neighbor),
                            context_type="neighbor",
                            relation_type="NEXT_CHUNK" if neighbor_key == "next_chunk_id" else "PREV_CHUNK",
                            retrieval_score=float(seed.get("score") or 0.0),
                        )
                    )

        priority = {"seed": 0, "parent": 1, "explicit_reference": 2, "cross_domain": 3, "neighbor": 4}
        results.sort(key=lambda item: (priority.get(str(item["context_type"]), 9), -float(item.get("score") or 0.0)))
        return results[:budget]


def _cli() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Expand legal retrieval context around seed chunks.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    retriever = HybridRetriever()
    seed = retriever.search(args.query, top_k=args.top_k)
    route = route_query(args.query, seed_chunks=seed)
    contexts = ContextExpander(retriever=retriever).expand(query=args.query, route_result=route, seed_chunks=seed)
    print(json.dumps({"route": route, "contexts": contexts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
