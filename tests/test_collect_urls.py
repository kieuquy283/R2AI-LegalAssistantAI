"""
Minimal unit tests for collect_urls extraction logic.

Run:
    python -m unittest tests.test_collect_urls
"""

import unittest

from src.ingestion.collect_urls import (
    build_paginated_urls,
    canonicalize_detail_url,
    deduplicate_records,
    extract_links_from_html,
    extract_next_page_links,
    is_luatvietnam_search_page,
)
from src.ingestion.source_registry import SourceConfig


class TestCollectUrls(unittest.TestCase):
    def make_source(self, *, provider="Test"):
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
            provider=provider,
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

    def test_is_luatvietnam_search_page(self):
        self.assertTrue(
            is_luatvietnam_search_page(
                "https://luatvietnam.vn/van-ban/tim-van-ban.html?keywords=luat+doanh+nghiep&PageIndex=2"
            )
        )
        self.assertFalse(
            is_luatvietnam_search_page(
                "https://luatvietnam.vn/tin-van-ban-moi/tiep-tuc-ra-soat-article.html"
            )
        )

    def test_extract_next_page_links_ignores_news_articles(self):
        html = """
        <html>
          <body>
            <a href="/van-ban/tim-van-ban.html?keywords=luat+doanh+nghiep&PageIndex=2">Tiếp</a>
            <a href="/tin-van-ban-moi/tiep-tuc-ra-soat-mien-giam-phi-article.html">Tiếp</a>
            <a href="/doanh-nghiep/luat-doanh-nghiep-2020-186272-d1.html">Tiếp</a>
          </body>
        </html>
        """
        source = self.make_source(provider="LuatVietnam")
        links = extract_next_page_links(html, source, source.url)
        self.assertEqual(
            links,
            ["https://luatvietnam.vn/van-ban/tim-van-ban.html?keywords=luat+doanh+nghiep&PageIndex=2"],
        )


    def test_build_paginated_urls_for_luatvietnam_detect_or_manual(self):
        raw = {
            "crawl_strategy": {
                "collect_document_links": True,
                "link_pattern": "-d1.html",
                "pagination": {
                    "enabled": True,
                    "strategy": "detect_or_manual",
                    "max_pages": 3,
                },
            }
        }
        source = SourceConfig(
            id="luatvietnam_search",
            name="LuatVietnam Search",
            provider="LuatVietnam",
            source_type="search_page",
            domain="business_law",
            url="https://luatvietnam.vn/van-ban/tim-van-ban.html?keywords=luat+doanh+nghiep",
            enabled=True,
            priority="high",
            crawl_frequency="weekly",
            raw=raw,
        )
        self.assertEqual(
            build_paginated_urls(source),
            [
                "https://luatvietnam.vn/van-ban/tim-van-ban.html?keywords=luat+doanh+nghiep",
                "https://luatvietnam.vn/van-ban/tim-van-ban.html?keywords=luat+doanh+nghiep&PagSize=20&PageSize=20&PageIndex=2",
                "https://luatvietnam.vn/van-ban/tim-van-ban.html?keywords=luat+doanh+nghiep&PagSize=20&PageSize=20&PageIndex=3",
            ],
        )
        self.assertEqual(
            build_paginated_urls(source, max_pages_override=4),
            [
                "https://luatvietnam.vn/van-ban/tim-van-ban.html?keywords=luat+doanh+nghiep",
                "https://luatvietnam.vn/van-ban/tim-van-ban.html?keywords=luat+doanh+nghiep&PagSize=20&PageSize=20&PageIndex=2",
                "https://luatvietnam.vn/van-ban/tim-van-ban.html?keywords=luat+doanh+nghiep&PagSize=20&PageSize=20&PageIndex=3",
                "https://luatvietnam.vn/van-ban/tim-van-ban.html?keywords=luat+doanh+nghiep&PagSize=20&PageSize=20&PageIndex=4",
            ],
        )


if __name__ == "__main__":
    unittest.main()
