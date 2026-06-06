import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.query_analytics import QueryAnalytics
from src.evaluation.user_feedback import UserFeedbackStore


class TestQueryAnalytics(unittest.TestCase):
    def test_analyze_logs_and_store_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            logs_dir = base / "eval_runs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "sample.jsonl").write_text(
                json.dumps(
                    {
                        "question": "Ai khong duoc thanh lap doanh nghiep?",
                        "route": "PARENT_CONTEXT",
                        "domains": ["business_law"],
                        "final_context_ids": ["c1"],
                        "citations": [{"citation": "Dieu 17"}],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = QueryAnalytics().analyze_eval_logs(logs_dir)
            self.assertIn("total_queries", report)
            self.assertEqual(report["total_queries"], 1)

            feedback_path = base / "feedback.jsonl"
            store = UserFeedbackStore(path=feedback_path)
            record = store.add_feedback(question="q", answer="a", rating="like", comment="ok")
            self.assertTrue(record["feedback_id"])
            self.assertEqual(len(store.list_feedback()), 1)


if __name__ == "__main__":
    unittest.main()
