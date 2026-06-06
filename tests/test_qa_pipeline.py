from __future__ import annotations

import unittest

from src.qa_pipeline import LegalQAPipeline


class LegalQAPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = LegalQAPipeline()

    def test_answer_returns_response_object(self) -> None:
        result = self.pipeline.answer("Ai khong duoc thanh lap doanh nghiep?")
        self.assertEqual(result["question"], "Ai khong duoc thanh lap doanh nghiep?")
        self.assertTrue(result["route"])
        self.assertTrue(result["answer"])
        self.assertTrue(result["final_contexts"])
        self.assertIn("citations", result)


if __name__ == "__main__":
    unittest.main()
