from __future__ import annotations

import unittest
from unittest.mock import patch

from src.qa_pipeline import LegalQAPipeline


class _LowConfidenceRetrieval:
    def run(self, query: str) -> dict:
        return {
            "route": "PARENT_CONTEXT",
            "domains": ["business_law", "labor_law"],
            "seed_chunks": [],
            "seed_contexts": [],
            "expanded_contexts": [],
            "final_contexts": [
                {
                    "chunk_id": "bad",
                    "content": "Van ban khong lien quan den cau hoi.",
                    "score": 0.01,
                    "final_score": 0.01,
                    "lexical_overlap": 0.0,
                    "title_match": 0.0,
                    "domain_match": 0.0,
                    "metadata": {
                        "doc_title": "Van ban sai chu de",
                        "citation": "Van ban sai chu de, Dieu 1",
                        "source_url": "https://example.com/sai",
                    },
                }
            ],
        }


class QAFallbackCitationGuardTests(unittest.TestCase):
    @patch("src.qa_pipeline.RetrievalPipeline", return_value=_LowConfidenceRetrieval())
    def test_low_confidence_contexts_do_not_attach_citations(self, _mock_retrieval) -> None:
        result = LegalQAPipeline().answer("giu ban chinh van bang chung chi cua nguoi lao dong bi xu phat the nao", use_llm=False)
        self.assertTrue(result["low_confidence"])
        self.assertEqual(result["final_contexts"], [])
        self.assertEqual(result["relevant_docs"], [])
        self.assertEqual(result["relevant_articles"], [])
        self.assertIn("Ch", result["answer"])


if __name__ == "__main__":
    unittest.main()
