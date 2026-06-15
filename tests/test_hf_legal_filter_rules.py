from __future__ import annotations

import unittest

from src.ingestion.hf_legal_filter_rules import evaluate_hf_legal_filter


class HFLegalFilterRulesTests(unittest.TestCase):
    def test_matches_luat_doanh_nghiep(self) -> None:
        result = evaluate_hf_legal_filter({"title": "Luật Doanh nghiệp năm 2020", "content": ""})
        self.assertTrue(result["include"])
        self.assertEqual(result["matched_group"], "business_sme")

    def test_matches_dnnvv(self) -> None:
        result = evaluate_hf_legal_filter({"title": "Hỗ trợ doanh nghiệp nhỏ và vừa", "content": ""})
        self.assertTrue(result["include"])
        self.assertIn("sme_support", result["candidate_domains"])

    def test_matches_nghi_dinh_80(self) -> None:
        result = evaluate_hf_legal_filter({"title": "Nghị định 80/2021/NĐ-CP", "content": ""})
        self.assertTrue(result["include"])
        self.assertEqual(result["matched_group"], "business_sme")

    def test_matches_hoa_don_dien_tu(self) -> None:
        result = evaluate_hf_legal_filter({"title": "Quy định về hóa đơn điện tử", "content": ""})
        self.assertTrue(result["include"])
        self.assertEqual(result["matched_group"], "tax_invoice_accounting")

    def test_matches_bhxh(self) -> None:
        result = evaluate_hf_legal_filter({"title": "Chậm đóng BHXH bắt buộc", "content": ""})
        self.assertTrue(result["include"])
        self.assertEqual(result["matched_group"], "labor_bhxh_union")

    def test_matches_so_huu_tri_tue(self) -> None:
        result = evaluate_hf_legal_filter({"title": "Xâm phạm quyền sở hữu trí tuệ đối với nhãn hiệu", "content": ""})
        self.assertTrue(result["include"])
        self.assertEqual(result["matched_group"], "intellectual_property")

    def test_excludes_internal_admin_noise(self) -> None:
        result = evaluate_hf_legal_filter({"title": "Quyết định bổ nhiệm thành lập ban chỉ đạo", "content": ""})
        self.assertFalse(result["include"])
        self.assertEqual(result["matched_group"], "excluded_noise")


if __name__ == "__main__":
    unittest.main()
