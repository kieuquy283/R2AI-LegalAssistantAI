from __future__ import annotations

import unittest

from rag.config.runtime import RetrievalRuntimeConfig
from src.retrieval.hybrid_fusion import select_dynamic_contexts


class DynamicContextSelectionTests(unittest.TestCase):
    def test_selection_varies_by_score_and_caps_levels(self) -> None:
        config = RetrievalRuntimeConfig(
            min_contexts=1,
            max_contexts=4,
            max_docs=1,
            max_articles=2,
            absolute_score_threshold=0.45,
            relative_score_threshold=0.75,
        )
        selected = select_dynamic_contexts(
            [
                {"retrieval_level": "doc", "doc_id": "d1", "final_score": 0.9},
                {"retrieval_level": "doc", "doc_id": "d1", "final_score": 0.88},
                {"retrieval_level": "article", "doc_id": "d1", "article": "Dieu 1", "final_score": 0.82},
                {"retrieval_level": "article", "doc_id": "d1", "article": "Dieu 2", "final_score": 0.79},
                {"retrieval_level": "article", "doc_id": "d1", "article": "Dieu 3", "final_score": 0.78},
                {"retrieval_level": "chunk", "doc_id": "d1", "final_score": 0.4},
            ],
            config=config,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(sum(1 for item in selected if item["retrieval_level"] == "doc"), 1)
        self.assertEqual(sum(1 for item in selected if item["retrieval_level"] == "article"), 2)


if __name__ == "__main__":
    unittest.main()
