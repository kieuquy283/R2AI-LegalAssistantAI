"""
Minimal unit tests for crawl_documents parsing logic.

Run:
    python -m unittest tests.test_crawl_documents
"""

import unittest

from src.ingestion.crawl_documents import (
    clean_markdown_noise,
    detect_access_restriction,
    extract_document_metadata,
    slugify,
)


class TestCrawlDocuments(unittest.TestCase):
    def test_slugify_vietnamese(self):
        self.assertEqual(
            slugify("Luật Doanh nghiệp 2020 Số 59/2020/QH14"),
            "luat_doanh_nghiep_2020_so_59_2020_qh14",
        )

    def test_extract_metadata(self):
        html = """
        <html>
          <head><title>Luật Doanh nghiệp 2020</title></head>
          <body>
            <h1>Luật Doanh nghiệp 2020</h1>
            <div>Số hiệu: 59/2020/QH14</div>
            <div>Loại văn bản: Luật</div>
            <div>Cơ quan ban hành: Quốc hội</div>
            <div>Ngày ban hành: 17/06/2020</div>
            <div>Ngày có hiệu lực: 01/01/2021</div>
            <div>Tình trạng hiệu lực: Còn hiệu lực</div>
          </body>
        </html>
        """
        metadata = extract_document_metadata(html, "https://example.com/doc")
        self.assertEqual(metadata["doc_title"], "Luật Doanh nghiệp 2020")
        self.assertEqual(metadata["doc_number"], "59/2020/QH14")
        self.assertEqual(metadata["doc_type"], "Luật")
        self.assertEqual(metadata["issuing_body"], "Quốc hội")

    def test_detect_access_restriction(self):
        result = detect_access_restriction(
            "<html>Vui lòng đăng nhập thành viên để xem chi tiết</html>",
            "",
        )
        self.assertTrue(result["login_required"])
        self.assertTrue(result["has_restriction_signal"])

    def test_clean_markdown_noise(self):
        md = "Nội dung chính\n\nVui lòng đăng nhập thành viên\n\nTin liên quan\n\nĐiều 1. Test"
        cleaned = clean_markdown_noise(md)
        self.assertIn("Nội dung chính", cleaned)
        self.assertIn("Điều 1. Test", cleaned)
        self.assertNotIn("Vui lòng đăng nhập", cleaned)
        self.assertNotIn("Tin liên quan", cleaned)


if __name__ == "__main__":
    unittest.main()
