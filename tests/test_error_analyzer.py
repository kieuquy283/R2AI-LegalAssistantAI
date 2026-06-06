import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.error_analyzer import ErrorAnalyzer


class TestErrorAnalyzer(unittest.TestCase):
    def test_analyze_file_counts_missing_citation_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            sample = base / "sample.jsonl"
            sample.write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "question": "test",
                        "route": "SIMPLE_VECTOR",
                        "final_context_ids": [],
                        "citations": [],
                        "answer": "test answer",
                        "grounding": {"has_citation": False, "is_grounded": False},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = ErrorAnalyzer().analyze_file(sample)
            self.assertEqual(report["total_records"], 1)
            self.assertGreaterEqual(report["missing_citation"], 1)
            self.assertGreaterEqual(report["no_context"], 1)


if __name__ == "__main__":
    unittest.main()
