from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.evaluate_qa import evaluate_questions, load_questions


class _FakeQAPipeline:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {
                "route": "PARENT_CONTEXT",
                "domains": ["business_law"],
                "answer": "1. Ket luan ngan: Co can cu.\n2. Can cu phap luat: Dieu 17.",
                "citations": [
                    {
                        "doc_title": "Luat Doanh nghiep 2020",
                        "doc_id": "59/2020/QH14",
                        "article": "Dieu 17",
                        "clause": None,
                        "citation": "Luat Doanh nghiep 2020, Dieu 17",
                        "source_url": "https://example.com/luat-doanh-nghiep",
                    }
                ],
                "relevant_doc_details": [
                    {
                        "doc_id": "59/2020/QH14",
                        "doc_title": "Luat Doanh nghiep 2020",
                        "source_url": "https://example.com/luat-doanh-nghiep",
                        "citation": "Luat Doanh nghiep 2020, Dieu 17",
                    }
                ],
                "relevant_article_details": [
                    {
                        "doc_id": "59/2020/QH14",
                        "doc_title": "Luat Doanh nghiep 2020",
                        "article": "Dieu 17",
                        "clause": None,
                        "citation": "Luat Doanh nghiep 2020, Dieu 17",
                        "source_url": "https://example.com/luat-doanh-nghiep",
                    }
                ],
                "final_contexts": [
                    {
                        "chunk_id": "c1",
                        "content": "Dieu 17. Nguoi khong duoc thanh lap doanh nghiep bao gom ...",
                        "metadata": {
                            "doc_id": "59/2020/QH14",
                            "doc_title": "Luat Doanh nghiep 2020",
                            "article": "Dieu 17",
                            "clause": None,
                            "citation": "Luat Doanh nghiep 2020, Dieu 17",
                            "source_url": "https://example.com/luat-doanh-nghiep",
                        },
                    }
                ],
                "seed_contexts": [],
                "expanded_contexts": [],
                "retrieved_chunks": [],
                "grounding": {"is_grounded": True},
            }

        return {
            "route": "SIMPLE_VECTOR",
            "domains": ["business_law"],
            "answer": "Khong co du lieu.",
            "citations": [],
            "relevant_doc_details": [],
            "relevant_article_details": [],
            "final_contexts": [],
            "seed_contexts": [],
            "expanded_contexts": [],
            "retrieved_chunks": [],
            "grounding": {"is_grounded": False},
        }


class EvaluateQAExportResultsTests(unittest.TestCase):
    def test_reads_jsonl_and_exports_results_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            questions_path = base / "questions.jsonl"
            output_path = base / "results.json"
            questions_path.write_text(
                "\n".join(
                    [
                        json.dumps({"question_id": "1", "question": "Question 1"}, ensure_ascii=False),
                        json.dumps({"id": "2", "question": "Question 2"}, ensure_ascii=False),
                        json.dumps({"question_id": "3", "question": "Question 3"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            questions = load_questions(questions_path)
            self.assertEqual(len(questions), 3)

            with patch("src.evaluation.evaluate_qa.LegalQAPipeline", return_value=_FakeQAPipeline()):
                summary = evaluate_questions(questions, run_id="unit_export_results", output_path=output_path, limit=3)

            self.assertEqual(summary["total_questions"], 3)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, list)
            self.assertEqual(len(payload), 3)
            for item in payload:
                self.assertIn("id", item)
                self.assertIn("question", item)
                self.assertIn("answer", item)
                self.assertIn("relevant_docs", item)
                self.assertIn("relevant_articles", item)
                self.assertIsInstance(item["relevant_docs"], list)
                self.assertIsInstance(item["relevant_articles"], list)

            self.assertEqual(payload[0]["id"], 1)
            self.assertEqual(payload[1]["id"], 2)
            self.assertEqual(payload[2]["id"], 3)
            self.assertTrue(payload[0]["relevant_docs"])
            self.assertTrue(payload[0]["relevant_articles"])
            self.assertEqual(payload[0]["relevant_docs"][0], "59/2020/QH14|Luat Doanh nghiep 2020")
            self.assertEqual(payload[0]["relevant_articles"][0], "59/2020/QH14|Luat Doanh nghiep 2020|Dieu 17")
            self.assertEqual(payload[1]["relevant_docs"], [])
            self.assertEqual(payload[1]["relevant_articles"], [])

    def test_missing_question_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "questions.jsonl"
            path.write_text(json.dumps({"id": "1"}), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_questions(path)

            self.assertIn("missing question text", str(ctx.exception))

    def test_exports_refs_from_top_level_context_fields_when_metadata_missing(self) -> None:
        class _TopLevelContextQAPipeline:
            def answer(self, question: str) -> dict:
                return {
                    "route": "PARENT_CONTEXT",
                    "domains": ["tax_law"],
                    "answer": "Có căn cứ về hóa đơn điện tử có mã của cơ quan thuế.",
                    "citations": [],
                    "relevant_doc_details": [],
                    "relevant_article_details": [],
                    "final_contexts": [
                        {
                            "chunk_id": "ctx-1",
                            "doc_id": "119/2018/NĐ-CP",
                            "doc_title": "Nghị định 119/2018/NĐ-CP quy định về hóa đơn điện tử khi bán hàng hóa, cung cấp dịch vụ",
                            "article": "Điều 19",
                            "citation": "Điều 19",
                            "metadata": {},
                        }
                    ],
                    "seed_contexts": [],
                    "expanded_contexts": [],
                    "retrieved_chunks": [],
                    "grounding": {"is_grounded": True},
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            output_path = base / "results.json"
            questions = [{"id": 1, "question": "hóa đơn điện tử có mã của cơ quan thuế"}]

            with patch("src.evaluation.evaluate_qa.LegalQAPipeline", return_value=_TopLevelContextQAPipeline()):
                evaluate_questions(questions, run_id="unit_top_level_context_refs", output_path=output_path, limit=1)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload[0]["relevant_docs"],
                ["119/2018/NĐ-CP|Nghị định 119/2018/NĐ-CP quy định về hóa đơn điện tử khi bán hàng hóa, cung cấp dịch vụ"],
            )
            self.assertEqual(
                payload[0]["relevant_articles"],
                [
                    "119/2018/NĐ-CP|Nghị định 119/2018/NĐ-CP quy định về hóa đơn điện tử khi bán hàng hóa, cung cấp dịch vụ|Điều 19"
                ],
            )


if __name__ == "__main__":
    unittest.main()
