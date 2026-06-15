from __future__ import annotations

import unittest
from unittest.mock import patch

from qdrant_client import QdrantClient

from rag.config.runtime import RetrievalRuntimeConfig
from src.ingestion.qdrant_index_builder import build_qdrant_index
from src.retrieval.qdrant_store import QdrantStore


class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class QdrantIndexBuilderTests(unittest.TestCase):
    @patch("src.ingestion.qdrant_index_builder.get_embeddings", return_value=_FakeEmbeddings())
    @patch("src.ingestion.qdrant_index_builder.read_jsonl")
    @patch("src.ingestion.qdrant_index_builder.QdrantStore")
    def test_build_qdrant_index_inserts_all_levels(self, store_cls, read_jsonl_mock, _embeddings_mock) -> None:
        read_jsonl_mock.side_effect = [
            [{"doc_id": "doc-1", "doc_title": "Doc", "doc_number": "01", "doc_type": "Luat", "issuer": "QH", "domain": "business_law", "cleaned_text": "abc"}],
            [{"node_id": "node-1", "doc_id": "doc-1", "title": "Điều 1", "article": "Điều 1", "content": "abc", "domain": "business_law"}],
            [{"chunk_id": "chunk-1", "doc_id": "doc-1", "doc_title": "Doc", "citation": "Doc, Điều 1", "content": "abc", "domain": "business_law"}],
        ]
        config = RetrievalRuntimeConfig(
            qdrant_collection_docs="legal_docs",
            qdrant_collection_articles="legal_articles",
            qdrant_collection_chunks="legal_chunks",
            qdrant_vector_size=4,
        )
        store = QdrantStore(client=QdrantClient(":memory:"), config=config)
        store_cls.return_value = store

        report = build_qdrant_index(documents_path="d", articles_path="a", chunks_path="c", recreate=True)
        self.assertEqual(report["documents_inserted"], 1)
        self.assertEqual(report["articles_inserted"], 1)
        self.assertEqual(report["chunks_inserted"], 1)
        self.assertEqual(report["legal_docs_count"], 1)


if __name__ == "__main__":
    unittest.main()
