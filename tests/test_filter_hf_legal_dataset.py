from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ingestion.filter_hf_legal_dataset import run_filter


class FilterHFLegalDatasetTests(unittest.TestCase):
    def test_run_filter_writes_normalized_schema(self) -> None:
        rows = [
            {
                "source_id": "1",
                "doc_id": "doc_1",
                "doc_title": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
                "doc_type": "Luật",
                "doc_number": "04/2017/QH14",
                "issuer": "Quốc hội",
                "issued_date": "2017-06-12",
                "effective_date": "2018-01-01",
                "source_url": "",
                "content": "Doanh nghiệp nhỏ và vừa được hỗ trợ theo luật.",
                "content_hash": "hash-1",
            },
            {
                "source_id": "1b",
                "doc_id": "doc_1_dup",
                "doc_title": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
                "doc_type": "Luật",
                "doc_number": "04/2017/QH14",
                "issuer": "Quốc hội",
                "issued_date": "2017-06-12",
                "effective_date": "2018-01-01",
                "source_url": "",
                "content": "Doanh nghiệp nhỏ và vừa được hỗ trợ theo luật.",
                "content_hash": "hash-1",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "filtered.jsonl"
            with patch("src.ingestion.filter_hf_legal_dataset._join_metadata_and_content", return_value=iter(rows)):
                report = run_filter("dummy-dataset", output_path)

            self.assertEqual(report["total_scanned_records"], 2)
            self.assertEqual(report["total_matched_records"], 2)
            self.assertEqual(report["total_deduplicated_records"], 1)
            output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(output_rows), 1)
            self.assertEqual(output_rows[0]["source_dataset"], "dummy-dataset")
            self.assertEqual(output_rows[0]["domain"], "sme_support")
            self.assertIn("candidate_domains", output_rows[0])


if __name__ == "__main__":
    unittest.main()
