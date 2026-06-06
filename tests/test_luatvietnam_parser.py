import unittest
from pathlib import Path

from src.ingestion.provider_parsers.luatvietnam_parser import parse_luatvietnam_metadata


FIXTURE_HTML = """
<html>
  <head>
    <title>Luat Doanh nghiep 2020, so 59/2020/QH14</title>
    <meta name="description" content="Luat Doanh nghiep 2020 co hieu luc tu 01/01/2021.">
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Legislation",
          "name": "Luat Doanh nghiep 2020, so 59/2020/QH14",
          "legislationIdentifier": "59/2020/QH14",
          "legislationType": "Luat",
          "legislationPassedBy": "Quoc hoi",
          "datePublished": "2020-06-17"
        }
      ]
    }
    </script>
  </head>
  <body>
    <h1>Luat Doanh nghiep 2020, so 59/2020/QH14</h1>
    <table>
      <tr><td><strong>Co quan ban hanh:</strong></td><td>Quoc hoi</td><td><strong>So hieu:</strong></td><td>59/2020/QH14</td></tr>
      <tr><td><strong>Loai van ban:</strong></td><td>Luat</td><td><strong>Nguoi ky:</strong></td><td>Nguyen Thi Kim Ngan</td></tr>
      <tr><td><strong>Ngay ban hanh:</strong></td><td>17/06/2020</td><td><strong>Tinh trang hieu luc:</strong></td><td>Con hieu luc</td></tr>
    </table>
  </body>
</html>
"""


class TestLuatVietnamParser(unittest.TestCase):
    def test_parser_reads_core_metadata(self) -> None:
        metadata = parse_luatvietnam_metadata(FIXTURE_HTML, url="https://luatvietnam.vn/test")
        self.assertEqual(metadata["doc_number"], "59/2020/QH14")
        self.assertEqual(metadata["doc_type"], "Luat")
        self.assertEqual(metadata["issuing_body"], "Quoc hoi")
        self.assertEqual(metadata["signer"], "Nguyen Thi Kim Ngan")
        self.assertEqual(metadata["issue_date"], "17/06/2020")
        self.assertEqual(metadata["effective_date"], "01/01/2021")
        self.assertEqual(metadata["status"], "Con hieu luc")
        self.assertGreater(metadata["confidence"]["doc_number"], 0.0)

    def test_real_html_does_not_return_boilerplate_status_or_dates(self) -> None:
        html_files = list(Path("data/raw/html").glob("*.html"))
        self.assertTrue(html_files)
        html = html_files[0].read_text(encoding="utf-8", errors="ignore")
        metadata = parse_luatvietnam_metadata(html, url="https://luatvietnam.vn/test")
        bad_values = ["đăng nhập", "tải về", "xem thêm", "tin liên quan"]
        for key in ["issue_date", "effective_date", "status"]:
            value = str(metadata.get(key) or "").lower()
            self.assertFalse(any(token in value for token in bad_values), (key, value))


if __name__ == "__main__":
    unittest.main()
