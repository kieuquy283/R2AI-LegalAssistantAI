import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval.legal_graph_store import LegalGraphStore


class TestLegalGraphStore(unittest.TestCase):
    def test_load_and_query_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            nodes = [
                {"node_id": "doc-1", "node_type": "document", "doc_id": "doc-1", "domain": "business_law", "title": "Luat A", "content": "", "source_url": "u1", "metadata": {}},
                {"node_id": "article-1", "node_type": "article", "doc_id": "doc-1", "domain": "business_law", "title": "Dieu 1", "content": "Noi dung", "source_url": "u1", "metadata": {}},
                {"node_id": "domain:administrative_penalty", "node_type": "domain", "doc_id": None, "domain": "administrative_penalty", "title": "administrative_penalty", "content": "", "source_url": None, "metadata": {}},
            ]
            edges = [
                {"edge_id": "e1", "source_id": "article-1", "target_id": "doc-1", "target_domain": None, "relation_type": "HAS_PARENT", "ref_text": None, "confidence": 1.0, "metadata": {}},
                {"edge_id": "e2", "source_id": "doc-1", "target_id": "article-1", "target_domain": None, "relation_type": "HAS_CHILD", "ref_text": None, "confidence": 1.0, "metadata": {}},
                {"edge_id": "e3", "source_id": "article-1", "target_id": "domain:administrative_penalty", "target_domain": "administrative_penalty", "relation_type": "CROSS_DOMAIN", "ref_text": None, "confidence": 0.7, "metadata": {}},
                {"edge_id": "e4", "source_id": "article-1", "target_id": "doc-1", "target_domain": None, "relation_type": "REFERS_TO", "ref_text": "Dieu 2", "confidence": 1.0, "metadata": {}},
            ]
            (base / "nodes.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in nodes), encoding="utf-8")
            (base / "edges.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in edges), encoding="utf-8")

            store = LegalGraphStore(nodes_path=base / "nodes.jsonl", edges_path=base / "edges.jsonl")
            self.assertIsNotNone(store.get_node("article-1"))
            self.assertIsNotNone(store.get_parent("article-1"))
            self.assertTrue(store.get_children("doc-1"))
            self.assertIsInstance(store.get_neighbors("article-1", max_depth=1), list)
            cross_domain_edges = store.get_cross_domains("article-1")
            self.assertEqual(cross_domain_edges[0]["target_domain"], "administrative_penalty")
            self.assertTrue(store.get_explicit_refs("article-1"))


if __name__ == "__main__":
    unittest.main()
