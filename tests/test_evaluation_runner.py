from __future__ import annotations

import json
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.evaluation.eval_logger import EvalLogger
from src.evaluation.evaluate_qa import evaluate_questions, load_questions


class EvaluationRunnerTests(unittest.TestCase):
    def test_eval_logger_writes_jsonl(self) -> None:
        logger = EvalLogger("unit_test_eval")
        logger.log(
            {
                "question_id": "q1",
                "question": "test",
                "route": "SIMPLE_VECTOR",
                "domains": ["business_law"],
                "answer": "test answer",
            }
        )
        self.assertTrue(logger.path.exists())
        self.assertGreater(logger.path.stat().st_size, 0)

    def test_evaluate_questions_writes_summary(self) -> None:
        questions = load_questions("data/evaluation/sample_questions.jsonl")
        self.assertTrue(questions)
        with patch.dict(os.environ, {"RETRIEVAL_BACKEND": "faiss", "EMBEDDING_BACKEND": "hash"}, clear=False):
            summary = evaluate_questions(questions[:1], run_id="unit_eval_summary")
        self.assertEqual(summary["total_questions"], 1)
        self.assertGreater(summary["answer_non_empty_rate"], 0)
        summary_path = Path("logs/eval_runs/unit_eval_summary_summary.json")
        self.assertTrue(summary_path.exists())
        parsed = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["total_questions"], 1)

    def test_load_questions_supports_json_array_and_exports_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "questions.json"
            output_path = Path(tmpdir) / "answers.json"
            input_path.write_text(
                json.dumps(
                    [
                        {"id": "q1", "question": "test question one"},
                        {"id": "q2", "question": "test question two"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            questions = load_questions(input_path)
            self.assertEqual(len(questions), 2)
            with patch.dict(os.environ, {"RETRIEVAL_BACKEND": "faiss", "EMBEDDING_BACKEND": "hash"}, clear=False):
                summary = evaluate_questions(questions[:1], run_id="unit_eval_answers", output_path=output_path)
            self.assertEqual(summary["total_questions"], 1)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, list)
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["id"], 1)
            self.assertEqual(payload[0]["question"], "test question one")
            self.assertIn("answer", payload[0])
            self.assertIn("relevant_docs", payload[0])
            self.assertIn("relevant_articles", payload[0])


if __name__ == "__main__":
    unittest.main()
