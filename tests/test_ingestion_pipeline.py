import unittest

from src.ingestion.document_parser import build_document_record
from src.ingestion.legal_chunker import build_chunks
from src.ingestion.legal_structure_parser import parse_document_structure
from src.ingestion.text_cleaner import clean_markdown_text


class TestIngestionPipeline(unittest.TestCase):
    def test_cleaner_keeps_legal_body(self):
        text = """# Luật Test
Ngày cập nhật: 01/01/2024
## TÓM TẮT LUẬT
Nội dung giới thiệu

**Chương I**
**Điều 1. Phạm vi điều chỉnh**
1. Khoản một
a) Điểm a
"""
        cleaned = clean_markdown_text(text)
        self.assertIn("Điều 1. Phạm vi điều chỉnh", cleaned)
        self.assertNotIn("TÓM TẮT", cleaned)

    def test_structure_parser_detects_article_clause_point(self):
        document = {
            "doc_id": "doc-1",
            "domain": "business_law",
            "source_url": "https://example.com",
            "cleaned_text": "Chương I\nĐiều 1. Phạm vi điều chỉnh\n1. Khoản một\na) Điểm a",
        }
        nodes = parse_document_structure(document)
        levels = {node["level"] for node in nodes}
        self.assertIn("article", levels)
        self.assertIn("clause", levels)
        self.assertIn("point", levels)

    def test_document_parser_cleans_manifest_boilerplate(self):
        manifest_row = {
            "success": True,
            "doc_id": "doc-1",
            "doc_title": "Luật A",
            "doc_number": "01/2024/QH15",
            "doc_type": "Luật",
            "issuing_body": "Quốc hội",
            "signer": "Nguyễn Văn A",
            "issue_date": "Ngày ban hành là ngày, tháng, năm văn bản được thông qua hoặc ký ban hành. 01/01/2024",
            "effective_date": "VB liên quan",
            "status": "Cho biết trạng thái hiệu lực của văn bản đang tra cứu",
            "domain": "business_law",
            "source_url": "https://example.com",
            "raw_html_path": "missing.html",
            "markdown_path": "doc.md",
            "content_hash": "abc",
        }
        row = build_document_record(manifest_row)
        self.assertEqual(row["issue_date"], "01/01/2024")
        self.assertIsNone(row["effective_date"])
        self.assertIsNone(row["status"])

    def test_chunker_assigns_context_and_links(self):
        documents = [
            {
                "doc_id": "doc-1",
                "doc_title": "Luật A",
                "domain": "business_law",
                "source_url": "https://example.com",
            }
        ]
        nodes = [
            {
                "node_id": "article-1",
                "doc_id": "doc-1",
                "level": "article",
                "article": "Điều 1",
                "article_title": "Phạm vi điều chỉnh",
                "clause": None,
                "point": None,
                "content": "Điều 1. Phạm vi điều chỉnh",
                "parent_id": None,
                "start_char": 0,
                "domain": "business_law",
                "source_url": "https://example.com",
            },
            {
                "node_id": "clause-1",
                "doc_id": "doc-1",
                "level": "clause",
                "article": "Điều 1",
                "article_title": "Phạm vi điều chỉnh",
                "clause": "Khoản 1",
                "point": None,
                "content": "1. Khoản một",
                "parent_id": "article-1",
                "start_char": 10,
                "domain": "business_law",
                "source_url": "https://example.com",
            },
        ]
        chunks, context_chunks, _ = build_chunks(nodes, documents)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(context_chunks), 1)
        self.assertTrue(chunks[0]["context_chunk_id"])


if __name__ == "__main__":
    unittest.main()
