import unittest

from src.generation.answer_generator import AnswerGenerator


class TestAnswerGenerator(unittest.TestCase):
    def test_answer_has_structure_and_citations(self):
        retrieval_result = {
            "final_contexts": [
                {
                    "chunk_id": "c1",
                    "content": "Người không được thành lập doanh nghiệp gồm cán bộ, công chức...",
                    "metadata": {
                        "doc_title": "Luật Doanh nghiệp 2020",
                        "article": "Điều 17",
                        "citation": "Luật Doanh nghiệp 2020, Điều 17",
                        "source_url": "https://example.com",
                    },
                }
            ]
        }
        result = AnswerGenerator().generate(query="Ai không được thành lập doanh nghiệp?", retrieval_result=retrieval_result)
        self.assertTrue(result["answer"])
        self.assertIn("Căn cứ pháp luật", result["answer"])
        self.assertTrue(result["citations"])

    def test_empty_context_reports_insufficient_basis(self):
        result = AnswerGenerator().generate(query="test", retrieval_result={"final_contexts": []})
        self.assertIn("Chưa đủ căn cứ", result["answer"])
