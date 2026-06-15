from __future__ import annotations

import unittest

from qdrant_client import QdrantClient

from rag.config.runtime import RetrievalRuntimeConfig
from src.retrieval.qdrant_store import QdrantStore


class QdrantStoreTests(unittest.TestCase):
    def test_upsert_rows_writes_points(self) -> None:
        config = RetrievalRuntimeConfig(qdrant_vector_size=4)
        store = QdrantStore(client=QdrantClient(":memory:"), config=config)
        store.recreate_collection("legal_docs")
        inserted = store.upsert_rows(
            collection_name="legal_docs",
            rows=[{"doc_id": "doc-1", "vector": [0.1, 0.2, 0.3, 0.4], "source_dataset": "hf"}],
            vector_key="vector",
            id_key="doc_id",
        )
        self.assertEqual(inserted, 1)
        self.assertEqual(store.count("legal_docs"), 1)


if __name__ == "__main__":
    unittest.main()
