import unittest

from src.retrieval.confidence_checker import ConfidenceChecker


class TestConfidenceChecker(unittest.TestCase):
    def setUp(self) -> None:
        self.checker = ConfidenceChecker()

    def test_low_confidence_penalty_query_recommends_cross_domain(self) -> None:
        result = self.checker.check(
            query="Khong gop du von dieu le dung han thi bi phat gi?",
            route_result={"route": "SIMPLE_VECTOR", "domains": ["business_law"]},
            seed_chunks=[
                {
                    "chunk_id": "c1",
                    "score": 0.42,
                    "content": "Dieu 47. Gop von thanh lap cong ty.",
                    "metadata": {
                        "domain": "business_law",
                        "source_url": "https://example.com",
                        "citation": "Luat Doanh nghiep, Dieu 47",
                        "article": "Dieu 47",
                    },
                }
            ],
        )
        self.assertIn("confidence_score", result)
        self.assertIn("should_escalate", result)
        self.assertFalse(result["is_confident"])
        self.assertTrue(result["should_escalate"])
        self.assertEqual(result["recommended_route"], "CROSS_DOMAIN_CONTEXT")

    def test_strong_seed_set_stays_confident(self) -> None:
        result = self.checker.check(
            query="Ai khong duoc thanh lap doanh nghiep?",
            route_result={"route": "PARENT_CONTEXT", "domains": ["business_law"]},
            seed_chunks=[
                {
                    "chunk_id": "c1",
                    "score": 0.82,
                    "content": "Dieu 17. To chuc, ca nhan khong co quyen thanh lap doanh nghiep.",
                    "metadata": {
                        "domain": "business_law",
                        "source_url": "https://example.com/1",
                        "citation": "Luat Doanh nghiep, Dieu 17",
                        "article": "Dieu 17",
                    },
                },
                {
                    "chunk_id": "c2",
                    "score": 0.74,
                    "content": "Khoan 2 Dieu 17 quy dinh cac doi tuong bi cam.",
                    "metadata": {
                        "domain": "business_law",
                        "source_url": "https://example.com/2",
                        "citation": "Luat Doanh nghiep, Khoan 2 Dieu 17",
                        "article": "Dieu 17",
                    },
                },
            ],
        )
        self.assertTrue(result["is_confident"])
        self.assertFalse(result["should_escalate"])
        self.assertEqual(result["recommended_route"], "PARENT_CONTEXT")


if __name__ == "__main__":
    unittest.main()
