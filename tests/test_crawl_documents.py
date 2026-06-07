"""
Targeted tests for crawl_documents.

Run:
    python -m unittest tests.test_crawl_documents
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.ingestion.crawl_documents import (
    clean_markdown_noise,
    crawl_detail_page,
    detect_access_restriction,
    extract_document_metadata,
    filter_resume_records,
    load_successful_crawl_keys,
    normalize_crawl_url_key,
    slugify,
)


class TestCrawlDocumentsUtils(unittest.TestCase):
    def test_slugify_ascii(self):
        self.assertEqual(
            slugify("Law on Enterprises 2020 No 59/2020/QH14"),
            "law_on_enterprises_2020_no_59_2020_qh14",
        )

    def test_extract_metadata_title(self):
        html = "<html><head><title>Law Title</title></head><body><h1>Law Title</h1></body></html>"
        metadata = extract_document_metadata(html, "https://example.com/doc")
        self.assertEqual(metadata["doc_title"], "Law Title")
        self.assertEqual(metadata["source_url"], "https://example.com/doc")

    def test_detect_access_restriction_ascii(self):
        result = detect_access_restriction(
            "<html>login required for premium members</html>",
            "",
        )
        self.assertTrue(result["login_required"])
        self.assertTrue(result["paywall_or_member_content"])
        self.assertTrue(result["has_restriction_signal"])

    def test_clean_markdown_noise_keeps_content(self):
        md = "# Law Title\n\nMain content\n\nRelated article\n\nArticle 1. Test"
        cleaned = clean_markdown_noise(md)
        self.assertIn("Main content", cleaned)
        self.assertIn("Article 1. Test", cleaned)

    def test_normalize_crawl_url_key_prefers_canonical_url(self):
        record = {
            "canonical_url": "https://example.com/doc-d1.html?utm_source=x#frag",
            "source_url": "https://example.com/ignored",
            "url": "https://example.com/ignored-too",
        }
        self.assertEqual(
            normalize_crawl_url_key(record),
            "https://example.com/doc-d1.html?utm_source=x",
        )

    def test_load_successful_crawl_keys_uses_success_rows_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"success": true, "canonical_url": "https://example.com/a-d1.html"}',
                        '{"success": false, "canonical_url": "https://example.com/b-d1.html"}',
                        '{"success": true, "source_url": "https://example.com/c-d1.html"}',
                    ]
                ),
                encoding="utf-8",
            )

            keys = load_successful_crawl_keys(path)
            self.assertEqual(
                keys,
                {
                    "https://example.com/a-d1.html",
                    "https://example.com/c-d1.html",
                },
            )

    def test_filter_resume_records_skips_success_but_keeps_failed(self):
        records = [
            {"canonical_url": "https://example.com/a-d1.html"},
            {"source_url": "https://example.com/b-d1.html"},
            {"url": "https://example.com/c-d1.html"},
        ]
        filtered, skipped = filter_resume_records(
            records,
            {"https://example.com/a-d1.html", "https://example.com/c-d1.html"},
        )
        self.assertEqual(skipped, 2)
        self.assertEqual(
            filtered,
            [{"source_url": "https://example.com/b-d1.html"}],
        )


class TestCrawlDocumentsRawHtml(unittest.IsolatedAsyncioTestCase):
    async def test_success_saves_result_html_and_required_fields(self):
        class FakeMarkdown:
            fit_markdown = "# Title\n\nBody"
            raw_markdown = ""

        class FakeResult:
            success = True
            html = "<html><body><h1>Law Title</h1><div>No: 59/2020/QH14</div></body></html>"
            markdown = FakeMarkdown()

        class FakeCrawler:
            async def arun(self, url, config):
                return FakeResult()

        record = {
            "url": "https://example.com/doc-d1.html",
            "canonical_url": "https://example.com/doc-d1.html",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            result = await crawl_detail_page(
                FakeCrawler(),
                record,
                html_dir=base / "html",
                markdown_dir=base / "markdown",
            )

            self.assertTrue(result["success"])
            self.assertIn("doc_id", result)
            self.assertIn("source_url", result)
            self.assertIn("raw_html_path", result)
            self.assertIn("html_hash", result)
            self.assertIn("crawl_time", result)

            html_path = Path(result["raw_html_path"])
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)
            self.assertEqual(html_path.read_text(encoding="utf-8"), FakeResult.html)
            self.assertEqual(
                result["html_hash"],
                hashlib.sha256(FakeResult.html.encode("utf-8")).hexdigest(),
            )

    async def test_failure_does_not_create_html_file(self):
        class FakeResult:
            success = False
            html = "<html>should not be written</html>"
            markdown = None
            error_message = "crawl failed"

        class FakeCrawler:
            async def arun(self, url, config):
                return FakeResult()

        record = {
            "url": "https://example.com/doc-d1.html",
            "canonical_url": "https://example.com/doc-d1.html",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            result = await crawl_detail_page(
                FakeCrawler(),
                record,
                html_dir=base / "html",
                markdown_dir=base / "markdown",
            )

            self.assertFalse(result["success"])
            self.assertNotIn("raw_html_path", result)
            self.assertFalse((base / "html").exists())


if __name__ == "__main__":
    unittest.main()
