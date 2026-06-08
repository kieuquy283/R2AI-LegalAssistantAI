from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.prepare_r2ai_dataset import prepare_r2ai_dataset


class PrepareR2aiDatasetTest(unittest.TestCase):
    def test_prepares_jsonl_from_stage1_payload(self) -> None:
        payload = [
            {
                "question_id": "q1",
                "current_question": "Ai khong duoc thanh lap doanh nghiep?",
                "expected_law_refs": ["Dieu 17"],
                "conversation": [{"role": "user", "text": "Hoi ve doanh nghiep"}],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "R2AIStage1DATA.json"
            output_path = Path(tmpdir) / "r2ai_stage1_questions.jsonl"
            input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            summary = prepare_r2ai_dataset(input_path=input_path, output_path=output_path)

            self.assertEqual(summary["items_read"], 1)
            self.assertEqual(summary["items_written"], 1)
            self.assertTrue(output_path.exists())

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], 1)
            self.assertEqual(rows[0]["question"], "Ai khong duoc thanh lap doanh nghiep?")
            self.assertEqual(rows[0]["question_id"], "q1")
            self.assertEqual(rows[0]["expected_law_refs"], ["Dieu 17"])
            self.assertEqual(rows[0]["conversation"][0]["content"], "Hoi ve doanh nghiep")


if __name__ == "__main__":
    unittest.main()
