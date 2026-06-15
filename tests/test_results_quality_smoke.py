from __future__ import annotations

import json
import unittest
from pathlib import Path


class ResultsQualitySmokeTests(unittest.TestCase):
    def test_results_fix_50_has_non_empty_answers_when_present(self) -> None:
        path = Path("results_fix_50.json")
        if not path.exists():
            self.skipTest("results_fix_50.json not found")
        rows = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 50)
        self.assertTrue(any(str(row.get("answer", "")).strip() for row in rows))
        self.assertFalse(all("Chưa đủ căn cứ pháp lý" in str(row.get("answer", "")) for row in rows))

    def test_fallback_rows_do_not_require_off_topic_docs(self) -> None:
        path = Path("results_fix_50.json")
        if not path.exists():
            self.skipTest("results_fix_50.json not found")
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            answer = str(row.get("answer", ""))
            if "Chưa đủ căn cứ pháp lý" not in answer:
                continue
            docs_text = " ".join(str(item) for item in row.get("relevant_docs", [])).lower()
            self.assertFalse(any(token in docs_text for token in ["xăng dầu", "y tế", "giáo dục", "ngân hàng"]))


if __name__ == "__main__":
    unittest.main()
