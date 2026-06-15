from __future__ import annotations

import unittest
from unittest.mock import patch

from src.retrieval.retrieval_pipeline import RetrievalPipeline


class _FakeRetriever:
    def search(self, query: str, *, top_k: int = 5, domain=None):
        return [
            {
                "chunk_id": "c1",
                "score": 0.83,
                "retrieval_score": 0.83,
                "content": "Điều 17. Người không được thành lập doanh nghiệp bao gồm ...",
                "metadata": {
                    "doc_id": "59/2020/QH14",
                    "doc_title": "Luật Doanh nghiệp 2020",
                    "article": "Điều 17",
                    "citation": "Luật Doanh nghiệp 2020, Điều 17",
                    "source_url": "https://example.com/luat-doanh-nghiep",
                    "domain": "business_law",
                },
            }
        ]


class _FakeExpander:
    def __init__(self, retriever=None):
        self.retriever = retriever

    def expand(self, *, query: str, route_result, seed_chunks):
        return []


class _FakeReranker:
    def rerank(self, query, expanded_contexts, max_contexts=5):
        return []


class RetrievalPipelineSmokeTests(unittest.TestCase):
    @patch("src.retrieval.retrieval_pipeline.Reranker", return_value=_FakeReranker())
    @patch("src.retrieval.retrieval_pipeline.ContextExpander", side_effect=lambda retriever=None: _FakeExpander(retriever=retriever))
    @patch("src.retrieval.retrieval_pipeline.HybridRetriever", return_value=_FakeRetriever())
    def test_final_contexts_stay_empty_when_reranker_rejects_all_contexts(self, *_mocks) -> None:
        pipeline = RetrievalPipeline()
        result = pipeline.run("Ai không được thành lập doanh nghiệp?")
        self.assertTrue(result["seed_contexts"])
        self.assertEqual(result["expanded_contexts"], [])
        self.assertEqual(result["final_contexts"], [])


if __name__ == "__main__":
    unittest.main()
