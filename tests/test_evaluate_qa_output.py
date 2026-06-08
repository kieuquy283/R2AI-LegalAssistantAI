from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.evaluate_qa import evaluate_questions


class _FakeQAPipeline:
    def answer(self, question: str) -> dict:
        if "fallback" in question:
            return {
                "route": "SIMPLE_VECTOR",
                "domains": ["business_law"],
                "answer": "Chưa đủ căn cứ pháp lý để kết luận.",
                "citations": [],
                "relevant_docs": [],
                "relevant_articles": [],
                "final_contexts": [],
                "seed_contexts": [],
                "expanded_contexts": [],
                "retrieved_chunks": [],
                "grounding": {"is_grounded": False},
            }
        return {
            "route": "PARENT_CONTEXT",
            "domains": ["business_law"],
            "answer": "1. Kết luận ngắn: Có căn cứ.\n2. Căn cứ pháp luật: Điều 17.",
            "citations": [
                {
                    "doc_title": "Luật Doanh nghiệp 2020",
                    "doc_id": "59/2020/QH14",
                    "article": "Điều 17",
                    "clause": None,
                    "citation": "Luật Doanh nghiệp 2020, Điều 17",
                    "source_url": "https://example.com/luat-doanh-nghiep",
                }
            ],
            "relevant_docs": ["59/2020/QH14|Luật Doanh nghiệp 2020"],
            "relevant_articles": ["59/2020/QH14|Luật Doanh nghiệp 2020|Điều 17"],
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
            "seed_contexts": [],
            "expanded_contexts": [],
            "retrieved_chunks": [],
            "grounding": {"is_grounded": True},
        }


class EvaluateQAOutputTests(unittest.TestCase):
    def test_logs_include_context_and_not_all_fallback(self) -> None:
        run_id = "unit_test_eval_qa_output"
        log_path = Path("logs/eval_runs") / f"{run_id}.jsonl"
        summary_path = Path("logs/eval_runs") / f"{run_id}_summary.json"
        for path in (log_path, summary_path):
            if path.exists():
                path.unlink()

        questions = [
            {"id": 1, "question": "Ai không được thành lập doanh nghiệp?"},
            {"id": 2, "question": "fallback question"},
        ]
        with patch("src.evaluation.evaluate_qa.LegalQAPipeline", return_value=_FakeQAPipeline()):
            summary = evaluate_questions(questions, run_id=run_id)

        self.assertEqual(summary["total_questions"], 2)
        self.assertTrue(log_path.exists())
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 2)
        fallback = [row for row in rows if "Chưa đủ căn cứ pháp lý" in str(row.get("answer") or "")]
        with_context = [
            row
            for row in rows
            if row.get("final_contexts") or row.get("citations") or row.get("relevant_docs") or row.get("relevant_articles")
        ]
        self.assertLess(len(fallback), len(rows))
        self.assertGreater(len(with_context), 0)


if __name__ == "__main__":
    unittest.main()
