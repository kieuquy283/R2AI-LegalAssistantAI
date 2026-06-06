import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from src.retrieval.hybrid_retriever import HybridRetriever


class FakeEmbeddings:
    def embed_query(self, text: str):
        return [1.0, 0.0] if "doanh nghiệp" in text.lower() else [0.0, 1.0]


class TestHybridRetriever(unittest.TestCase):
    def test_search_returns_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            index = faiss.IndexFlatIP(2)
            vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
            faiss.normalize_L2(vectors)
            index.add(vectors)
            faiss.write_index(index, str(base / "faiss.index"))

            metadata = [
                {"index": 0, "chunk_id": "c1", "doc_id": "d1", "domain": "business_law", "source_url": "u1", "citation": "Điều 1"},
                {"index": 1, "chunk_id": "c2", "doc_id": "d2", "domain": "tax_law", "source_url": "u2", "citation": "Điều 2"},
            ]
            (base / "chunk_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            (base / "chunks.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"chunk_id": "c1", "doc_id": "d1", "domain": "business_law", "doc_title": "Luật A", "article": "Điều 1", "clause": None, "citation": "Luật A, Điều 1", "source_url": "u1", "content": "doanh nghiệp", "embedding_text": "doanh nghiệp"}, ensure_ascii=False),
                        json.dumps({"chunk_id": "c2", "doc_id": "d2", "domain": "tax_law", "doc_title": "Luật B", "article": "Điều 2", "clause": None, "citation": "Luật B, Điều 2", "source_url": "u2", "content": "thuế", "embedding_text": "thuế"}, ensure_ascii=False),
                    ]
                ),
                encoding="utf-8",
            )
            (base / "bm25_corpus.json").write_text(
                json.dumps(
                    [
                        {"chunk_id": "c1", "text": "doanh nghiệp", "tokens": ["doanh", "nghiệp"]},
                        {"chunk_id": "c2", "text": "thuế", "tokens": ["thuế"]},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            retriever = HybridRetriever(
                faiss_index_path=base / "faiss.index",
                metadata_path=base / "chunk_metadata.json",
                chunks_path=base / "chunks.jsonl",
                bm25_corpus_path=base / "bm25_corpus.json",
                embedding_model=FakeEmbeddings(),
            )
            results = retriever.search("Ai không được thành lập doanh nghiệp?", top_k=2)
            self.assertTrue(results)
            self.assertEqual(results[0]["chunk_id"], "c1")
            self.assertTrue(results[0]["content"])
            self.assertTrue(results[0]["metadata"]["source_url"])

    def test_domain_filter_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            index = faiss.IndexFlatIP(2)
            vectors = np.array([[1.0, 0.0]], dtype="float32")
            faiss.normalize_L2(vectors)
            index.add(vectors)
            faiss.write_index(index, str(base / "faiss.index"))
            (base / "chunk_metadata.json").write_text(json.dumps([{"index": 0, "chunk_id": "c1", "doc_id": "d1", "domain": "business_law", "source_url": "u1", "citation": "Điều 1"}], ensure_ascii=False), encoding="utf-8")
            (base / "chunks.jsonl").write_text(json.dumps({"chunk_id": "c1", "doc_id": "d1", "domain": "business_law", "doc_title": "Luật A", "article": "Điều 1", "clause": None, "citation": "Luật A, Điều 1", "source_url": "u1", "content": "doanh nghiệp", "embedding_text": "doanh nghiệp"}, ensure_ascii=False), encoding="utf-8")
            retriever = HybridRetriever(
                faiss_index_path=base / "faiss.index",
                metadata_path=base / "chunk_metadata.json",
                chunks_path=base / "chunks.jsonl",
                embedding_model=FakeEmbeddings(),
            )
            self.assertEqual(retriever.search("test", top_k=5, domain="tax_law"), [])
