from __future__ import annotations

import unittest

from src.ingestion.common import slugify_vi, strip_vietnamese_accents


class TestVietnameseSlugify(unittest.TestCase):
    def test_strip_accents(self):
        self.assertEqual(strip_vietnamese_accents("Hướng dẫn"), "Huong dan")
        self.assertEqual(strip_vietnamese_accents("Điều 4"), "Dieu 4")
        self.assertEqual(strip_vietnamese_accents("Cơ sở ươm tạo"), "Co so uom tao")

    def test_long_title(self):
        slug = slugify_vi(
            "Hướng dẫn việc thành lập cơ sở ươm tạo doanh nghiệp nhỏ và vừa"
        )
        self.assertEqual(
            slug,
            "huong_dan_viec_thanh_lap_co_so_uom_tao_doanh_nghiep_nho_va_vua",
        )

    def test_luat_doanh_nghiep(self):
        self.assertEqual(slugify_vi("Luật Doanh nghiệp"), "luat_doanh_nghiep")

    def test_nghi_dinh_hoa_don(self):
        self.assertEqual(
            slugify_vi("Nghị định về hóa đơn, chứng từ"),
            "nghi_dinh_ve_hoa_don_chung_tu",
        )

    def test_bo_luat_lao_dong(self):
        self.assertEqual(slugify_vi("Bộ luật Lao động"), "bo_luat_lao_dong")

    def test_cong_ty_tnhh_mot_thanh_vien(self):
        self.assertEqual(
            slugify_vi("Công ty trách nhiệm hữu hạn một thành viên"),
            "cong_ty_trach_nhiem_huu_han_mot_thanh_vien",
        )

    def test_empty_input(self):
        self.assertEqual(slugify_vi(""), "unknown")

    def test_only_punctuation(self):
        self.assertEqual(slugify_vi("!!! ___ ..."), "unknown")

    def test_with_numbers(self):
        self.assertEqual(
            slugify_vi("Luật số 04/2017/QH14"),
            "luat_so_04_2017_qh14",
        )

    def test_truncated_long_title(self):
        title = "Hướng dẫn " * 20
        slug = slugify_vi(title, max_length=50)
        self.assertLessEqual(len(slug), 50)
        self.assertTrue(slug.endswith("huong_dan") or slug.endswith("_huong"))

    def test_no_trailing_underscore(self):
        slug = slugify_vi("Hướng dẫn !!! ___")
        self.assertFalse(slug.endswith("_"))

    def test_no_leading_underscore(self):
        slug = slugify_vi("!!! ___ Hướng dẫn")
        self.assertFalse(slug.startswith("_"))

    def test_strip_vietnamese_accents_preserves_case(self):
        result = strip_vietnamese_accents("Hướng Dẫn Việc")
        self.assertEqual(result, "Huong Dan Viec")

    def test_slugify_vi_lowercases(self):
        slug = slugify_vi("Hướng Dẫn")
        self.assertEqual(slug, "huong_dan")

    def test_doc_id_stays_ascii(self):
        expected = slugify_vi("159227")
        self.assertEqual(expected, "159227")

    def test_strip_vietnamese_accents_dd(self):
        self.assertEqual(strip_vietnamese_accents("đồng"), "dong")
        self.assertEqual(strip_vietnamese_accents("Đồng"), "Dong")

    def test_slugify_vi_mixed_content(self):
        slug = slugify_vi(
            "Nghị định 78/2021/TT-BTC về hóa đơn, chứng từ"
        )
        self.assertEqual(
            slug,
            "nghi_dinh_78_2021_tt_btc_ve_hoa_don_chung_tu",
        )


if __name__ == "__main__":
    unittest.main()
