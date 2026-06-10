from __future__ import annotations

import unittest
from unittest.mock import patch

from src.qa_pipeline import LegalQAPipeline


DOC_ID = "luat_doanh_nghiep_2020_so_59_2020_qh14_59_2020_qh14"


class _FakeRetrievalPipeline:
    def run(self, query: str) -> dict:
        context = {
            "chunk_id": "c1",
            "content": "Điều 17. Người không được thành lập doanh nghiệp bao gồm cán bộ, công chức và một số chủ thể khác theo luật.",
            "metadata": {
                "doc_id": DOC_ID,
                "doc_title": "Luật Doanh nghiệp 2020, số 59/2020/QH14",
                "article": "Điều 17",
                "clause": None,
                "citation": "Luật Doanh nghiệp 2020, số 59/2020/QH14, Điều 17",
                "source_url": "https://example.com/luat-doanh-nghiep",
            },
        }
        return {
            "route": "PARENT_CONTEXT",
            "domains": ["business_law"],
            "seed_chunks": [context],
            "seed_contexts": [context],
            "expanded_contexts": [context],
            "final_contexts": [context],
        }


class QAPipelineSmokeTests(unittest.TestCase):
    @patch("src.qa_pipeline.RetrievalPipeline", return_value=_FakeRetrievalPipeline())
    def test_answer_includes_reference_fields_in_submission_format(self, _mock_retrieval) -> None:
        qa = LegalQAPipeline()
        result = qa.answer("Ai không được thành lập doanh nghiệp?")
        self.assertTrue(result["answer"])
        self.assertTrue(result["final_contexts"])
        self.assertTrue(result["citations"])
        self.assertTrue(result["relevant_docs"])
        self.assertTrue(result["relevant_articles"])
        self.assertNotIn("Căn cứ pháp luật", result["answer"])
        self.assertEqual(result["relevant_docs"][0], "59/2020/QH14|Luật 59/2020/QH14 Doanh nghiệp 2020")
        self.assertEqual(result["relevant_articles"][0], "59/2020/QH14|Luật 59/2020/QH14 Doanh nghiệp 2020|Điều 17")


if __name__ == "__main__":
    unittest.main()
