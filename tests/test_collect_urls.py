"""
Minimal unit tests for collect_urls extraction logic.

Run:
    python -m unittest tests.test_collect_urls
"""

import unittest

from src.ingestion.collect_urls import (
    canonicalize_detail_url,
    deduplicate_records,
    extract_links_from_html,
)
from src.ingestion.source_registry import SourceConfig


class TestCollectUrls(unittest.TestCase):
    def make_source(self):
        raw = {
            "crawl_strategy": {
                "collect_document_links": True,
                "link_pattern": "-d1.html",
                "pagination": {"enabled": False},
            }
        }
        return SourceConfig(
            id="test_source",
            name="Test Source",
            provider="Test",
            source_type="search_page",
            domain="business_law",
            url="https://luatvietnam.vn/van-ban/tim-van-ban.html",
            enabled=True,
            priority="high",
            crawl_frequency="weekly",
            raw=raw,
        )

    def test_extract_detail_links(self):
        html = """
        <html>
          <body>
            <a href="/doanh-nghiep/luat-doanh-nghiep-2020-186272-d1.html">Luật DN</a>
            <a href="/tin-tuc/not-doc.html">Tin tức</a>
            <a href="https://external.com/doc-d1.html">External</a>
            <a href="/doanh-nghiep/luat-doanh-nghiep-2020-186272-d1.html">Duplicate</a>
          </body>
        </html>
        """
        source = self.make_source()
        records = extract_links_from_html(html, source, source.url)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["url"].endswith("-d1.html"))
        self.assertEqual(records[0]["domain"], "business_law")

    def test_canonicalize_removes_tracking_params(self):
        url = "https://example.com/a-d1.html?utm_source=x&doc=1&fbclid=abc"
        canonical = canonicalize_detail_url(url)
        self.assertIn("doc=1", canonical)
        self.assertNotIn("utm_source", canonical)
        self.assertNotIn("fbclid", canonical)

    def test_deduplicate_records(self):
        records = [
            {"url": "https://x.com/a-d1.html?utm_source=x", "canonical_url": "https://x.com/a-d1.html"},
            {"url": "https://x.com/a-d1.html", "canonical_url": "https://x.com/a-d1.html"},
            {"url": "https://x.com/b-d1.html", "canonical_url": "https://x.com/b-d1.html"},
        ]
        unique = deduplicate_records(records)
        self.assertEqual(len(unique), 2)
        self.assertTrue(all("url_hash" in r for r in unique))


if __name__ == "__main__":
    unittest.main()
