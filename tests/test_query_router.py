import unittest

from src.retrieval.query_router import route_query


class TestQueryRouter(unittest.TestCase):
    def test_routes_match_expected_patterns(self):
        cases = [
            ("Cong ty TNHH mot thanh vien la gi?", {"business_law"}),
            ("Ai khong duoc thanh lap doanh nghiep?", {"business_law"}),
            ("Khong gop du von dieu le dung han thi bi phat gi?", {"business_law", "administrative_penalty"}),
            ("Cong ty moi thanh lap can lam nhung viec gi?", {"business_law"}),
            ("Nguoi nuoc ngoai gop von vao cong ty Viet Nam can dieu kien gi?", {"business_law", "investment_law"}),
        ]
        for query, expected_domains in cases:
            result = route_query(query)
            self.assertTrue(result["route"])
            self.assertTrue(expected_domains.issubset(set(result["domains"])))
