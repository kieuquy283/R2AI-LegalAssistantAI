from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.retrieval.retrieval_pipeline import RetrievalPipeline


class _FakeDenseRetriever:
    def search(self, query, preferred_domains=None):
        return [{"candidate_id": "chunk:c1", "retrieval_level": "chunk", "chunk_id": "c1", "doc_id": "d1", "doc_title": "Doc", "citation": "Doc, Dieu 1", "article": "Dieu 1", "domain": "business_law", "content": "abc", "dense_score": 0.9, "metadata": {"doc_id": "d1", "doc_title": "Doc", "article": "Dieu 1", "citation": "Doc, Dieu 1", "domain": "business_law"}}]


class _FakeSparseRetriever:
    def search(self, query, top_k, preferred_domains=None):
        return []


class _FakeExactSearch:
    def search(self, query, top_k, preferred_domains=None):
        return []


class BackendSelectionTests(unittest.TestCase):
    @patch("src.retrieval.retrieval_pipeline.LegalExactSearch", return_value=_FakeExactSearch())
    @patch("src.retrieval.retrieval_pipeline.BM25Retriever", return_value=_FakeSparseRetriever())
    @patch("src.retrieval.retrieval_pipeline.QdrantRetriever", return_value=_FakeDenseRetriever())
    def test_qdrant_backend_uses_qdrant_path(self, *_mocks) -> None:
        previous = os.environ.get("RETRIEVAL_BACKEND")
        os.environ["RETRIEVAL_BACKEND"] = "qdrant"
        try:
            pipeline = RetrievalPipeline()
            result = pipeline.run("doanh nghiep nho va vua")
        finally:
            if previous is None:
                os.environ.pop("RETRIEVAL_BACKEND", None)
            else:
                os.environ["RETRIEVAL_BACKEND"] = previous
        self.assertIn(result["route"], {"SIMPLE_VECTOR", "PARENT_CONTEXT"})
        self.assertTrue(result["final_contexts"])


if __name__ == "__main__":
    unittest.main()
