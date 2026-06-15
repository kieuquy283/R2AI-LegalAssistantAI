from __future__ import annotations

import unittest

from src.retrieval.query_router import CROSS_DOMAIN_CONTEXT, PARENT_CONTEXT, route_query


class DomainRouterTests(unittest.TestCase):
    def test_dnnvv_query_stays_single_family(self) -> None:
        result = route_query("doanh nghiep nho va vua duoc ho tro tu van nhu the nao")
        self.assertIn("business_law", result["domains"])
        self.assertNotEqual(result["route"], CROSS_DOMAIN_CONTEXT)

    def test_invoice_query_prefers_parent_context(self) -> None:
        result = route_query("hoa don dien tu co ma cua co quan thue")
        self.assertIn("tax_law", result["domains"])
        self.assertEqual(result["route"], PARENT_CONTEXT)

    def test_bhxh_query_is_not_forced_cross_domain(self) -> None:
        result = route_query("cham dong bao hiem xa hoi bat buoc bi phat the nao")
        self.assertIn("social_insurance", result["domains"])
        self.assertNotEqual(result["route"], CROSS_DOMAIN_CONTEXT)

    def test_ip_certificate_query_prefers_ip_domain(self) -> None:
        result = route_query("van bang bao ho so huu cong nghiep tham dinh noi dung cap khi nao")
        self.assertIn("ip_law", result["domains"])
        self.assertNotIn("labor_law", result["domains"])
        self.assertEqual(result["route"], PARENT_CONTEXT)

    def test_explicit_multi_domain_query_uses_cross_domain(self) -> None:
        result = route_query("nhuong quyen thuong mai logistics va xuat nhap khau can dieu kien gi")
        self.assertEqual(result["route"], CROSS_DOMAIN_CONTEXT)


if __name__ == "__main__":
    unittest.main()
