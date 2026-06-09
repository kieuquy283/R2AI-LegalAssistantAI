from __future__ import annotations

import unittest
from unittest.mock import patch

from src.qa_pipeline import LegalQAPipeline


class _FakeRetrievalPipeline:
    def run(self, query: str) -> dict:
        return {
            "route": "PARENT_CONTEXT",
            "domains": ["business_law"],
            "seed_chunks": [
                {
                    "chunk_id": "c1",
                    "content": "Điều 17. Người không được thành lập doanh nghiệp bao gồm ...",
                    "metadata": {
                        "doc_id": "59/2020/QH14",
                        "doc_title": "Luật Doanh nghiệp 2020",
                        "article": "Điều 17",
                        "clause": None,
                        "citation": "Luật Doanh nghiệp 2020, Điều 17",
                        "source_url": "https://example.com/luat-doanh-nghiep",
                    },
                }
            ],
            "seed_contexts": [
                {
                    "chunk_id": "c1",
                    "content": "Điều 17. Người không được thành lập doanh nghiệp bao gồm ...",
                    "metadata": {
                        "doc_id": "59/2020/QH14",
                        "doc_title": "Luật Doanh nghiệp 2020",
                        "article": "Điều 17",
                        "clause": None,
                        "citation": "Luật Doanh nghiệp 2020, Điều 17",
                        "source_url": "https://example.com/luat-doanh-nghiep",
                    },
                }
            ],
            "expanded_contexts": [
                {
                    "chunk_id": "c1",
                    "content": "Điều 17. Người không được thành lập doanh nghiệp bao gồm ...",
                    "metadata": {
                        "doc_id": "59/2020/QH14",
                        "doc_title": "Luật Doanh nghiệp 2020",
                        "article": "Điều 17",
                        "clause": None,
                        "citation": "Luật Doanh nghiệp 2020, Điều 17",
                        "source_url": "https://example.com/luat-doanh-nghiep",
                    },
                }
            ],
            "final_contexts": [
                {
                    "chunk_id": "c1",
                    "content": "Điều 17. Người không được thành lập doanh nghiệp bao gồm ...",
                    "metadata": {
                        "doc_id": "59/2020/QH14",
                        "doc_title": "Luật Doanh nghiệp 2020",
                        "article": "Điều 17",
                        "clause": None,
                        "citation": "Luật Doanh nghiệp 2020, Điều 17",
                        "source_url": "https://example.com/luat-doanh-nghiep",
                    },
                }
            ],
        }


class QAPipelineSmokeTests(unittest.TestCase):
    @patch("src.qa_pipeline.RetrievalPipeline", return_value=_FakeRetrievalPipeline())
    def test_answer_includes_citations_and_reference_fields(self, _mock_retrieval) -> None:
        qa = LegalQAPipeline()
        result = qa.answer("Ai không được thành lập doanh nghiệp?")
        self.assertTrue(result["answer"])
        self.assertTrue(result["final_contexts"])
        self.assertTrue(result["citations"])
        self.assertTrue(result["relevant_docs"])
        self.assertTrue(result["relevant_articles"])
        self.assertNotIn("Chưa đủ căn cứ pháp lý", result["answer"])
        self.assertEqual(result["relevant_docs"][0], "59/2020/QH14|Luật Doanh nghiệp 2020")
        self.assertEqual(result["relevant_articles"][0], "59/2020/QH14|Luật Doanh nghiệp 2020|Điều 17")


if __name__ == "__main__":
    unittest.main()
