from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from src.retrieval.hybrid_retriever import HybridRetriever


class _FlatEmbeddings:
    def embed_query(self, text: str):
        return [1.0, 0.0]


class RetrievalRelevanceSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        base = Path(self._tmpdir.name)

        index = faiss.IndexFlatIP(2)
        vectors = np.array([[1.0, 0.0]] * 4, dtype="float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)
        faiss.write_index(index, str(base / "faiss.index"))

        metadata = [
            {"index": 0, "chunk_id": "c1", "doc_id": "d1", "domain": "business_law", "source_url": "u1", "citation": "Luat Ho tro doanh nghiep nho va vua 2017"},
            {"index": 1, "chunk_id": "c2", "doc_id": "d2", "domain": "labor_law", "source_url": "u2", "citation": "Bo luat Lao dong 2019"},
            {"index": 2, "chunk_id": "c3", "doc_id": "d3", "domain": "business_law", "source_url": "u3", "citation": "Luat Quan ly thue 2019"},
            {"index": 3, "chunk_id": "c4", "doc_id": "d4", "domain": "business_law", "source_url": "u4", "citation": "Luat So huu tri tue 2005"},
        ]
        (base / "chunk_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

        chunks = [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "domain": "business_law",
                "doc_title": "Luat Ho tro doanh nghiep nho va vua 2017",
                "article": "Dieu 4",
                "clause": None,
                "citation": "Luat Ho tro doanh nghiep nho va vua 2017, Dieu 4",
                "source_url": "u1",
                "content": "Co so uom tao doanh nghiep nho va vua, khu lam viec chung, ho tro doanh nghiep nho va vua.",
                "embedding_text": "Luat Ho tro doanh nghiep nho va vua 2017 co so uom tao doanh nghiep nho va vua khu lam viec chung.",
            },
            {
                "chunk_id": "c2",
                "doc_id": "d2",
                "domain": "labor_law",
                "doc_title": "Bo luat Lao dong 2019",
                "article": "Dieu 17",
                "clause": "Khoan 1",
                "citation": "Bo luat Lao dong 2019, Dieu 17, Khoan 1",
                "source_url": "u2",
                "content": "Giữ bản chính giấy tờ tùy thân, văn bằng, chứng chỉ của người lao động.",
                "embedding_text": "Bo luat Lao dong 2019 Dieu 17 Khoan 1 giu ban chinh giay to tuy than van bang chung chi cua nguoi lao dong.",
            },
            {
                "chunk_id": "c3",
                "doc_id": "d3",
                "domain": "business_law",
                "doc_title": "Luat Quan ly thue 2019",
                "article": "Dieu 88",
                "clause": "Khoan 3",
                "citation": "Luat Quan ly thue 2019, Dieu 88, Khoan 3",
                "source_url": "u3",
                "content": "Hoa don dien tu co ma cua co quan thue, chu ky so.",
                "embedding_text": "Luat Quan ly thue 2019 hoa don dien tu co ma cua co quan thue chu ky so.",
            },
            {
                "chunk_id": "c4",
                "doc_id": "d4",
                "domain": "business_law",
                "doc_title": "Luat So huu tri tue 2005",
                "article": "Dieu 50",
                "clause": None,
                "citation": "Luat So huu tri tue 2005, Dieu 50",
                "source_url": "u4",
                "content": "Ho so dang ky quyen tac gia bao gom giay to.",
                "embedding_text": "Luat So huu tri tue 2005 ho so dang ky quyen tac gia bao gom giay to.",
            },
        ]
        (base / "chunks.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in chunks), encoding="utf-8")

        bm25_rows = [
            {"chunk_id": "c1", "text": "Luat Ho tro doanh nghiep nho va vua 2017 co so uom tao doanh nghiep nho va vua khu lam viec chung", "tokens": []},
            {"chunk_id": "c2", "text": "Bo luat Lao dong 2019 giu ban chinh giay to tuy than van bang chung chi cua nguoi lao dong", "tokens": []},
            {"chunk_id": "c3", "text": "Luat Quan ly thue 2019 hoa don dien tu co ma cua co quan thue chu ky so", "tokens": []},
            {"chunk_id": "c4", "text": "Luat So huu tri tue 2005 ho so dang ky quyen tac gia bao gom giay to", "tokens": []},
        ]
        (base / "bm25_corpus.json").write_text(json.dumps(bm25_rows, ensure_ascii=False), encoding="utf-8")

        self.retriever = HybridRetriever(
            faiss_index_path=base / "faiss.index",
            metadata_path=base / "chunk_metadata.json",
            chunks_path=base / "chunks.jsonl",
            bm25_corpus_path=base / "bm25_corpus.json",
            embedding_model=_FlatEmbeddings(),
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _top_titles(self, query: str, top_k: int = 3) -> list[str]:
        return [str(row.get("metadata", {}).get("doc_title") or "") for row in self.retriever.search(query, top_k=top_k)]

    def test_dnnvv_query_prefers_sme_docs(self) -> None:
        titles = self._top_titles("doanh nghiep nho va vua ho tro co so uom tao khu lam viec chung thue dat dai")
        self.assertTrue(any("doanh nghiep nho va vua" in title.lower() for title in titles[:3]))

    def test_bhxh_query_prefers_labor_docs(self) -> None:
        titles = self._top_titles("cham dong bao hiem xa hoi bat buoc xu phat")
        self.assertTrue(any("bao hiem xa hoi" in title.lower() or "lao dong" in title.lower() for title in titles[:3]))
        self.assertFalse(any(token in title.lower() for token in ["xang dau", "y te", "giao duc"] for title in titles[:3]))

    def test_invoice_query_prefers_tax_docs(self) -> None:
        titles = self._top_titles("hoa don dien tu co ma co quan thue chu ky so")
        self.assertTrue(any("quan ly thue" in title.lower() or "hoa don dien tu" in title.lower() for title in titles[:3]))

    def test_ip_query_prefers_ip_docs(self) -> None:
        titles = self._top_titles("ho so dang ky quyen tac gia bao gom giay to gi")
        self.assertTrue(any("so huu tri tue" in title.lower() or "quyen tac gia" in title.lower() for title in titles[:3]))


if __name__ == "__main__":
    unittest.main()
