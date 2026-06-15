from __future__ import annotations

import unittest

from rag.config.runtime import RetrievalRuntimeConfig
from src.retrieval.qdrant_retriever import QdrantRetriever


class _FakeEmbeddings:
    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class _FakeHit:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class _FakeClient:
    def search(self, collection_name, query_vector, limit, with_payload, with_vectors):
        return [
            _FakeHit(
                0.91,
                {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "doc_title": "Luat Doanh nghiep",
                    "doc_number": "59/2020/QH14",
                    "article": "Dieu 1",
                    "citation": "Luat Doanh nghiep, Dieu 1",
                    "domain": "business_law",
                    "content": "abc",
                },
            )
        ]


class _FakeStore:
    def __init__(self):
        self.client = _FakeClient()


class QdrantRetrieverTests(unittest.TestCase):
    def test_search_returns_dense_candidates(self) -> None:
        retriever = QdrantRetriever(
            store=_FakeStore(),
            config=RetrievalRuntimeConfig(candidate_k_docs=1, candidate_k_articles=1, candidate_k_chunks=1),
            embeddings=_FakeEmbeddings(),
        )
        results = retriever.search("59/2020/QH14 Dieu 1", preferred_domains=["business_law"])
        self.assertTrue(results)
        self.assertEqual(results[0]["retrieval_method"], "dense")
        self.assertEqual(results[0]["doc_number"], "59/2020/QH14")


if __name__ == "__main__":
    unittest.main()
