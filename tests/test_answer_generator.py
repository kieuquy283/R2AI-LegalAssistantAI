import unittest

from src.generation.answer_generator import AnswerGenerator
from src.generation.prompt_builder import PromptBuilder


class TestAnswerGenerator(unittest.TestCase):
    def test_prompt_builder_contains_context_and_grounding_rules(self):
        builder = PromptBuilder()
        prompt = builder.build(
            query="Ai khong duoc thanh lap doanh nghiep?",
            contexts=[
                {
                    "chunk_id": "c1",
                    "content": "Dieu 17. To chuc, ca nhan co quyen thanh lap doanh nghiep, tru cac truong hop...",
                    "metadata": {
                        "doc_title": "Luat Doanh nghiep 2020",
                        "article": "Dieu 17",
                        "citation": "Luat Doanh nghiep 2020, Dieu 17",
                        "source_url": "https://example.com",
                        "domain": "business_law",
                    },
                }
            ],
            route="PARENT_CONTEXT",
            domains=["business_law"],
        )
        self.assertIn("CONTEXT", prompt["user_prompt"])
        self.assertIn("Không bịa", prompt["system_prompt"])
        self.assertIn("Căn cứ pháp luật", prompt["user_prompt"])

    def test_answer_has_structure_and_citations(self):
        retrieval_result = {
            "route": "PARENT_CONTEXT",
            "domains": ["business_law"],
            "final_contexts": [
                {
                    "chunk_id": "c1",
                    "content": "Nguoi khong duoc thanh lap doanh nghiep gom can bo, cong chuc...",
                    "metadata": {
                        "doc_title": "Luat Doanh nghiep 2020",
                        "article": "Dieu 17",
                        "citation": "Luat Doanh nghiep 2020, Dieu 17",
                        "source_url": "https://example.com",
                    },
                }
            ],
        }
        result = AnswerGenerator().generate(query="Ai khong duoc thanh lap doanh nghiep?", retrieval_result=retrieval_result)
        self.assertTrue(result["answer"])
        self.assertIn("Căn cứ pháp luật", result["answer"])
        self.assertTrue(result["citations"])
        self.assertEqual(result["generation_mode"], "template")

    def test_empty_context_reports_insufficient_basis(self):
        result = AnswerGenerator().generate(
            query="test",
            retrieval_result={"route": "SIMPLE_VECTOR", "domains": [], "final_contexts": []},
        )
        self.assertIn("Chưa đủ căn cứ", result["answer"])


if __name__ == "__main__":
    unittest.main()
