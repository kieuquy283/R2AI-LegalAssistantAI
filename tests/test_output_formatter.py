from __future__ import annotations

import unittest

from src.evaluation.output_formatter import format_article_ref, format_doc_ref, format_submission_record


class OutputFormatterTests(unittest.TestCase):
    def test_format_doc_ref_returns_pipe_string(self) -> None:
        ref = format_doc_ref(
            {
                "doc_id": "59/2020/QH14",
                "doc_title": "Luat 59/2020/QH14 Doanh nghiep",
                "citation": "Luat 59/2020/QH14, Dieu 17",
            }
        )
        self.assertEqual(ref, "59/2020/QH14|Luat 59/2020/QH14 Doanh nghiep")

    def test_format_article_ref_returns_article_string(self) -> None:
        ref = format_article_ref(
            {
                "doc_id": "59/2020/QH14",
                "doc_title": "Luat 59/2020/QH14 Doanh nghiep",
                "article": "Dieu 17",
            }
        )
        self.assertEqual(ref, "59/2020/QH14|Luat 59/2020/QH14 Doanh nghiep|Dieu 17")

    def test_format_submission_record_keeps_only_string_lists(self) -> None:
        record = format_submission_record(
            {
                "id": 1,
                "question": "Q",
                "answer": "A",
                "relevant_docs": ["59/2020/QH14|Luat Doanh nghiep", "bad"],
                "relevant_articles": ["59/2020/QH14|Luat Doanh nghiep|Dieu 17", "bad"],
            }
        )
        self.assertEqual(record["relevant_docs"], ["59/2020/QH14|Luat Doanh nghiep"])
        self.assertEqual(record["relevant_articles"], ["59/2020/QH14|Luat Doanh nghiep|Dieu 17"])


if __name__ == "__main__":
    unittest.main()
