from __future__ import annotations

import json
import unittest
from pathlib import Path

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
        summary = evaluate_questions(questions[:1], run_id="unit_eval_summary")
        self.assertEqual(summary["total_questions"], 1)
        self.assertGreater(summary["answer_non_empty_rate"], 0)
        summary_path = Path("logs/eval_runs/unit_eval_summary_summary.json")
        self.assertTrue(summary_path.exists())
        parsed = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["total_questions"], 1)


if __name__ == "__main__":
    unittest.main()
