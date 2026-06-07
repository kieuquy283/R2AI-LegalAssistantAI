import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion.validate_crawl_quality import validate_crawl_quality


class TestValidateCrawlQuality(unittest.TestCase):
    def test_validate_crawl_quality_writes_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            html = base / "doc.html"
            md = base / "doc.md"
            html.write_text("<html><body><h1>Luat</h1></body></html>", encoding="utf-8")
            md.write_text("# Điều 1\n\nNội dung pháp luật", encoding="utf-8")
            manifest = base / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "success": True,
                                "domain": "business_law",
                                "source_url": "https://example.com/doc",
                                "raw_html_path": str(html),
                                "markdown_path": str(md),
                                "access_restriction": {"has_restriction_signal": False},
                            },
                            ensure_ascii=False,
                        )
                        for _ in range(5)
                    ]
                ),
                encoding="utf-8",
            )

            report_path = base / "report.json"
            report = validate_crawl_quality(manifest_path=manifest, output_path=report_path, limit=5)

            self.assertTrue(report_path.exists())
            self.assertEqual(report["sampled_success_rows"], 5)
            self.assertEqual(report["missing_paths"], 0)
            self.assertGreater(report["legal_signal_count"], 0)


if __name__ == "__main__":
    unittest.main()
