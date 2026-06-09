from __future__ import annotations

import json
import unittest
from pathlib import Path


class ResultsExportQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results_path = Path("results_fix_50.json")
        cls.results = None
        if cls.results_path.exists():
            cls.results = json.loads(cls.results_path.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        if self.results is None:
            self.skipTest("results_fix_50.json not found")

    def test_export_is_list_and_has_expected_size(self) -> None:
        self.assertIsInstance(self.results, list)
        self.assertEqual(len(self.results), 50)

    def test_records_have_required_fields(self) -> None:
        for row in self.results[:10]:
            self.assertIn("id", row)
            self.assertIn("question", row)
            self.assertIn("answer", row)
            self.assertIn("relevant_docs", row)
            self.assertIn("relevant_articles", row)
            self.assertIsInstance(row["relevant_docs"], list)
            self.assertIsInstance(row["relevant_articles"], list)

    def test_answers_are_not_empty_or_all_fallback(self) -> None:
        empty_answers = [row for row in self.results if not str(row.get("answer", "")).strip()]
        fallback_answers = [row for row in self.results if "Chưa đủ căn cứ pháp lý" in str(row.get("answer", ""))]
        self.assertEqual(len(empty_answers), 0)
        self.assertLess(len(fallback_answers), len(self.results))

    def test_dnnvv_records_are_not_lawless_or_off_topic(self) -> None:
        hits = [row for row in self.results if "doanh nghiệp nhỏ và vừa" in str(row.get("question", "")).lower()]
        self.assertGreater(len(hits), 0)
        for row in hits[:5]:
            docs = " ".join(
                " ".join(str(item.get(key, "")) for key in ("doc_title", "citation", "source_url"))
                for item in row.get("relevant_docs", [])
            ).lower()
            self.assertTrue(
                any(token in docs for token in ["doanh nghiệp nhỏ và vừa", "hỗ trợ doanh nghiệp nhỏ và vừa", "dnnvv"]),
                msg=f"Off-topic docs for row {row.get('id')}: {docs[:500]}",
            )

    def test_no_obvious_off_topic_documents_in_sample(self) -> None:
        forbidden = ["xăng dầu", "y tế", "giáo dục"]
        sample = self.results[:20]
        for row in sample:
            docs = " ".join(
                " ".join(str(item.get(key, "")) for key in ("doc_title", "citation", "source_url"))
                for item in row.get("relevant_docs", [])
            ).lower()
            if any(token in str(row.get("question", "")).lower() for token in ["bảo hiểm xã hội", "hóa đơn", "quyền tác giả", "doanh nghiệp nhỏ và vừa"]):
                self.assertFalse(
                    any(token in docs for token in forbidden),
                    msg=f"Off-topic docs for row {row.get('id')}: {docs[:500]}",
                )


if __name__ == "__main__":
    unittest.main()
