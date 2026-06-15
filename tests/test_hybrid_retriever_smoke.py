from __future__ import annotations

import unittest

from rag.retrieval.vectorstore import OfflineHashEmbeddings
from src.retrieval.hybrid_retriever import HybridRetriever


class HybridRetrieverSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = HybridRetriever(embedding_model=OfflineHashEmbeddings())

    def test_returns_results_for_legal_query(self) -> None:
        results = self.retriever.search("Ai không được thành lập doanh nghiệp?", top_k=5)
        self.assertGreater(len(results), 0)
        first = results[0]
        self.assertTrue(first.get("chunk_id"))
        self.assertTrue(first.get("content"))


if __name__ == "__main__":
    unittest.main()
