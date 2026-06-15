from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion.normalize_hf_filtered_docs import normalize_hf_filtered_docs


class NormalizeHFFilteredDocsTests(unittest.TestCase):
    def test_normalizes_filtered_rows_into_documents_nodes_and_chunks(self) -> None:
        rows = [
            {
                "source_dataset": "hf-dataset",
                "source_id": "1",
                "doc_id": "doc-1",
                "doc_title": "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
                "doc_type": "Luật",
                "doc_number": "04/2017/QH14",
                "issuer": "Quốc hội",
                "issued_date": "2017-06-12",
                "effective_date": "2018-01-01",
                "domain": "sme_support",
                "candidate_domains": ["sme_support", "business_law"],
                "matched_group": "business_sme",
                "matched_keywords": ["doanh nghiep nho va vua"],
                "priority": 1,
                "source_url": "",
                "content": "Điều 4. Doanh nghiệp nhỏ và vừa là doanh nghiệp đáp ứng tiêu chí theo luật.\nĐiều 5. Nguyên tắc hỗ trợ doanh nghiệp nhỏ và vừa.",
                "content_hash": "hash-1",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_path = base / "filtered.jsonl"
            input_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            report = normalize_hf_filtered_docs(input_path, base / "hf_smoke")
            self.assertEqual(report["documents"], 1)
            self.assertGreater(report["legal_nodes"], 0)
            self.assertGreater(report["chunks"], 0)

            documents = [json.loads(line) for line in Path(report["documents_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
            chunks = [json.loads(line) for line in Path(report["chunks_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(documents[0]["doc_number"], "04/2017/QH14")
            self.assertTrue(any(chunk["citation"] for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
