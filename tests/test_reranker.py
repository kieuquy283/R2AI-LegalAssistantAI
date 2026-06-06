import unittest

from src.retrieval.reranker import Reranker


class TestReranker(unittest.TestCase):
    def test_reranker_prefers_grounded_seed(self):
        query = "Ai không được thành lập doanh nghiệp?"
        contexts = [
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
        ranked = Reranker().rerank(query, contexts, max_contexts=1)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["chunk_id"], "a")
        self.assertIn("final_score", ranked[0])
