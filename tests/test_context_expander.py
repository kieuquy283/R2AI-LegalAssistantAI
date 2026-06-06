import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval.context_expander import ContextExpander


class FakeRetriever:
    def search(self, query, top_k=5, domain=None):
        return []


class TestContextExpander(unittest.TestCase):
    def test_expand_parent_and_neighbor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            chunks = [
                {"chunk_id": "seed1", "doc_id": "d1", "domain": "business_law", "doc_title": "Luật A", "article": "Điều 1", "clause": None, "citation": "Luật A, Điều 1", "source_url": "u1", "content": "seed", "context_chunk_id": "ctx1", "prev_chunk_id": None, "next_chunk_id": "n1", "explicit_refs": []},
                {"chunk_id": "n1", "doc_id": "d1", "domain": "business_law", "doc_title": "Luật A", "article": "Điều 1", "clause": "Khoản 1", "citation": "Luật A, Điều 1, Khoản 1", "source_url": "u1", "content": "neighbor", "context_chunk_id": "ctx1", "prev_chunk_id": "seed1", "next_chunk_id": None, "explicit_refs": []},
            ]
            contexts = [{"context_chunk_id": "ctx1", "doc_id": "d1", "domain": "business_law", "doc_title": "Luật A", "article": "Điều 1", "citation": "Luật A, Điều 1", "source_url": "u1", "content": "parent"}]
            edges = [{"source_id": "seed1", "target_id": "ctx1", "relation_type": "HAS_PARENT", "confidence": 1.0}]
            cross_edges = [{"source_id": "seed1", "target_id": "domain:administrative_penalty", "relation_type": "RELATED_DOMAIN", "confidence": 0.8, "source_domain": "business_law", "target_domain": "administrative_penalty"}]
            (base / "chunks.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in chunks), encoding="utf-8")
            (base / "context_chunks.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in contexts), encoding="utf-8")
            (base / "legal_edges.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in edges), encoding="utf-8")
            (base / "cross_domain_edges.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in cross_edges), encoding="utf-8")

            expander = ContextExpander(
                chunks_path=base / "chunks.jsonl",
                context_chunks_path=base / "context_chunks.jsonl",
                edges_path=base / "legal_edges.jsonl",
                cross_domain_edges_path=base / "cross_domain_edges.jsonl",
                retriever=FakeRetriever(),
            )
            seed = [{"chunk_id": "seed1", "score": 0.8, "retrieval_score": 0.8, "content": "seed", "metadata": {"source_url": "u1", "domain": "business_law"}}]
            route = {"route": "MULTI_DOMAIN_COMPLEX", "domains": ["business_law", "administrative_penalty"], "needs_parent": True, "needs_neighbor": True, "needs_graph": False, "needs_cross_domain": True}
            expanded = expander.expand(query="Không góp đủ vốn điều lệ đúng hạn thì bị phạt gì?", route_result=route, seed_chunks=seed)
            ids = [item["chunk_id"] for item in expanded]
            self.assertIn("ctx1", ids)
            self.assertIn("n1", ids)
            self.assertLessEqual(len(expanded), 12)
