import unittest

from src.generation.grounding_validator import GroundingValidator


class TestGroundingValidator(unittest.TestCase):
    def test_grounded_answer_passes(self) -> None:
        validator = GroundingValidator()
        answer = "Can cu Luat Doanh nghiep 2020, Dieu 17, mot so doi tuong khong duoc thanh lap doanh nghiep."
        contexts = [
            {
                "chunk_id": "c1",
                "content": "Dieu 17. Quyen thanh lap doanh nghiep...",
                "metadata": {
                    "doc_title": "Luat Doanh nghiep 2020",
                    "article": "Dieu 17",
                    "source_url": "https://example.com",
                    "citation": "Luat Doanh nghiep 2020, Dieu 17",
                },
            }
        ]
        citations = [{"doc_title": "Luat Doanh nghiep 2020", "article": "Dieu 17", "source_url": "https://example.com"}]
        result = validator.validate(query="Ai khong duoc thanh lap doanh nghiep?", answer=answer, citations=citations, contexts=contexts)
        self.assertTrue(result["has_citation"])
        self.assertTrue(result["is_grounded"])

    def test_empty_context_requires_insufficient_basis_notice(self) -> None:
        validator = GroundingValidator()
        result = validator.validate(query="test", answer="Day la cau tra loi tu do.", citations=[], contexts=[])
        self.assertIn("empty_context_without_insufficient_basis_notice", result["warnings"])


if __name__ == "__main__":
    unittest.main()
