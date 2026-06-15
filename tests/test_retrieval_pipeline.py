import unittest
import os
from unittest.mock import patch

from src.retrieval.retrieval_pipeline import RetrievalPipeline


class _FakeRetriever:
    def search(self, query: str, *, top_k: int = 5, domain=None):
        return [
            {
                "chunk_id": "c1",
                "score": 0.35,
                "retrieval_score": 0.35,
                "content": "Dieu 47. Gop von thanh lap cong ty.",
                "metadata": {
                    "domain": "business_law",
                    "source_url": "https://example.com/1",
                    "citation": "Luat Doanh nghiep, Dieu 47",
                    "article": "Dieu 47",
                },
            }
        ]


class _FakeExpander:
    def __init__(self, retriever=None):
        self.retriever = retriever

    def expand(self, *, query: str, route_result, seed_chunks):
        return [
            {
                "chunk_id": "ctx-1",
                "content": "Ngu can cu bo sung ve xu phat hanh chinh.",
                "context_type": "cross_domain",
                "relation_type": "RELATED_DOMAIN",
                "score": 0.4,
                "retrieval_score": 0.4,
                "metadata": {"domain": "administrative_penalty", "citation": "Nghi dinh X"},
            }
        ] + list(seed_chunks)


class _FakeReranker:
    def rerank(self, query, expanded_contexts, max_contexts=5):
        return list(expanded_contexts)[:max_contexts]


class TestRetrievalPipeline(unittest.TestCase):
    @patch("src.retrieval.retrieval_pipeline.Reranker", return_value=_FakeReranker())
    @patch("src.retrieval.retrieval_pipeline.ContextExpander", side_effect=lambda retriever=None: _FakeExpander(retriever=retriever))
    @patch("src.retrieval.retrieval_pipeline.HybridRetriever", return_value=_FakeRetriever())
    def test_pipeline_includes_confidence_and_escalated_route(self, *_mocks) -> None:
        previous = os.environ.get("RETRIEVAL_BACKEND")
        os.environ["RETRIEVAL_BACKEND"] = "faiss"
        try:
            pipeline = RetrievalPipeline()
            result = pipeline.run("Khong gop du von dieu le dung han thi bi phat gi?")
        finally:
            if previous is None:
                os.environ.pop("RETRIEVAL_BACKEND", None)
            else:
                os.environ["RETRIEVAL_BACKEND"] = previous
        self.assertIn("confidence_result", result)
        self.assertFalse(result["confidence_result"]["is_confident"])
        self.assertEqual(result["route"], "CROSS_DOMAIN_CONTEXT")
        self.assertTrue(result["final_contexts"])


if __name__ == "__main__":
    unittest.main()
