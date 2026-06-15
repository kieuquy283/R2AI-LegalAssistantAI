from __future__ import annotations

import unittest

from src.retrieval.reranker import Reranker


class RerankerRelevanceTests(unittest.TestCase):
    def test_dnnvv_procurement_context_beats_admin_procedure_doc(self) -> None:
        query = "doanh nghiep nho va vua uu dai dau thau"
        contexts = [
            {
                "chunk_id": "good",
                "content": "Doanh nghiep nho va vua duoc huong ho tro va uu dai trong dau thau theo quy dinh.",
                "retrieval_score": 0.42,
                "context_type": "seed",
                "metadata": {
                    "doc_title": "Nghi dinh 80/2021/ND-CP huong dan thi hanh Luat Ho tro doanh nghiep nho va vua",
                    "citation": "Nghi dinh 80/2021/ND-CP, Dieu 24",
                    "source_url": "https://luatvietnam.vn/doanh-nghiep/nghi-dinh-80-2021-nd-cp-208340-d1.html",
                },
            },
            {
                "chunk_id": "bad",
                "content": "Cong bo thu tuc hanh chinh trong linh vuc thanh lap doanh nghiep, ho kinh doanh va dau thau.",
                "retrieval_score": 0.66,
                "context_type": "seed",
                "metadata": {
                    "doc_title": "Quyet dinh 855/QD-BKHDT ve thu tuc hanh chinh thanh lap doanh nghiep, ho kinh doanh",
                    "citation": "Quyet dinh 855/QD-BKHDT, Dieu 3",
                    "source_url": "https://luatvietnam.vn/doanh-nghiep/quyet-dinh-855-qd-bkhdt-ve-thu-tuc-hanh-chinh-thanh-lap-doanh-nghiep-ho-kinh-doanh-204696-d1.html",
                },
            },
        ]
        ranked = Reranker().rerank(query, contexts, max_contexts=2)
        self.assertEqual(ranked[0]["chunk_id"], "good")

    def test_invoice_context_beats_generic_tax_refund(self) -> None:
        query = "hoa don dien tu co ma cua co quan thue"
        contexts = [
            {
                "chunk_id": "good",
                "content": "Hoa don dien tu co ma cua co quan thue va chu ky so duoc quy dinh tai dieu 6.",
                "retrieval_score": 0.55,
                "context_type": "seed",
                "metadata": {
                    "doc_title": "Thong tu 68/2019/TT-BTC huong dan ve hoa don dien tu",
                    "citation": "Thong tu 68/2019/TT-BTC, Dieu 6",
                    "source_url": "https://luatvietnam.vn/ke-toan/thong-tu-68-2019-tt-btc-huong-dan-ve-hoa-don-dien-tu-177381-d1.html",
                },
            },
            {
                "chunk_id": "bad",
                "content": "Thu tuc hoan thue gia tri gia tang doi voi thiet bi nhap khau.",
                "retrieval_score": 0.7,
                "context_type": "seed",
                "metadata": {
                    "doc_title": "Thong tu 205/2009/TT-BTC huong dan thu tuc hoan thue gia tri gia tang",
                    "citation": "Thong tu 205/2009/TT-BTC, Dieu 2",
                    "source_url": "https://luatvietnam.vn/thue/thong-tu-205-2009-tt-btc-bo-tai-chinh-47239-d1.html",
                },
            },
        ]
        ranked = Reranker().rerank(query, contexts, max_contexts=2)
        self.assertEqual(ranked[0]["chunk_id"], "good")

    def test_social_insurance_penalty_beats_generic_labor_doc(self) -> None:
        query = "cham dong bao hiem xa hoi bat buoc xu phat"
        contexts = [
            {
                "chunk_id": "good",
                "content": "Cham dong bao hiem xa hoi bat buoc co the bi xu phat theo quy dinh ve bao hiem xa hoi.",
                "retrieval_score": 0.41,
                "context_type": "seed",
                "metadata": {
                    "doc_title": "Luat Bao hiem xa hoi 2024",
                    "citation": "Luat Bao hiem xa hoi 2024, Dieu 39",
                    "source_url": "https://luatvietnam.vn/bao-hiem/luat-bao-hiem-xa-hoi-2024-41-2024-qh15-385675-d1.html",
                },
            },
            {
                "chunk_id": "bad",
                "content": "Hop dong lao dong co the giao ket bang van ban hoac thong diep du lieu.",
                "retrieval_score": 0.69,
                "context_type": "seed",
                "metadata": {
                    "doc_title": "Nghi dinh 44/2013/ND-CP quy dinh ve hop dong lao dong",
                    "citation": "Nghi dinh 44/2013/ND-CP, Dieu 3",
                    "source_url": "https://luatvietnam.vn/lao-dong/nghi-dinh-44-2013-nd-cp-79107-d1.html",
                },
            },
        ]
        ranked = Reranker().rerank(query, contexts, max_contexts=2)
        self.assertEqual(ranked[0]["chunk_id"], "good")

    def test_ip_certificate_context_beats_labor_doc(self) -> None:
        query = "sau khi co thong bao ket qua tham dinh noi dung can lam gi de duoc cap van bang bao ho so huu cong nghiep"
        contexts = [
            {
                "chunk_id": "good",
                "content": "Sau khi có thông báo kết quả thẩm định nội dung, người nộp đơn phải nộp lệ phí cấp văn bằng bảo hộ.",
                "retrieval_score": 0.4,
                "context_type": "seed",
                "metadata": {
                    "doc_title": "Thong tu 23/2023/TT-BKHCN huong dan Luat So huu tri tue ve quyen so huu cong nghiep",
                    "citation": "Thong tu 23/2023/TT-BKHCN, Dieu 18",
                    "source_url": "https://luatvietnam.vn/so-huu-tri-tue/thong-tu-23-2023-tt-bkhcn-277152-d1.html",
                },
            },
            {
                "chunk_id": "bad",
                "content": "Nguoi su dung lao dong co nghia vu thong bao ket qua cho nguoi lao dong.",
                "retrieval_score": 0.63,
                "context_type": "seed",
                "metadata": {
                    "doc_title": "Bo luat Lao dong 2019",
                    "citation": "Bo luat Lao dong 2019, Dieu 4",
                    "source_url": "https://luatvietnam.vn/lao-dong/bo-luat-lao-dong-2019-so-45-2019-qh14-179015-d1.html",
                },
            },
        ]
        ranked = Reranker().rerank(query, contexts, max_contexts=2)
        self.assertEqual(ranked[0]["chunk_id"], "good")

    def test_ip_context_beats_unrelated_investment_doc(self) -> None:
        query = "ten thuong mai dieu kien bao ho so huu tri tue"
        contexts = [
            {
                "chunk_id": "good",
                "content": "Ten thuong mai duoc bao ho neu co kha nang phan biet va duoc su dung hop phap.",
                "retrieval_score": 0.45,
                "context_type": "seed",
                "metadata": {
                    "doc_title": "Luat So huu tri tue 2005",
                    "citation": "Luat So huu tri tue 2005, Dieu 76",
                    "source_url": "https://luatvietnam.vn/so-huu-tri-tue/luat-so-huu-tri-tue-2005-so-50-2005-qh11-18077-d1.html",
                },
            },
            {
                "chunk_id": "bad",
                "content": "Dau tu von nha nuoc vao doanh nghiep theo quy dinh moi.",
                "retrieval_score": 0.6,
                "context_type": "seed",
                "metadata": {
                    "doc_title": "Nghi dinh 32/2018/ND-CP sua doi quy dinh ve dau tu von Nha nuoc vao doanh nghiep",
                    "citation": "Nghi dinh 32/2018/ND-CP, Dieu 1",
                    "source_url": "https://luatvietnam.vn/tai-chinh/nghi-dinh-32-2018-nd-cp-sua-doi-bo-sung-nghi-dinh-91-2015-nd-cp-160566-d1.html",
                },
            },
        ]
        ranked = Reranker().rerank(query, contexts, max_contexts=2)
        self.assertEqual(ranked[0]["chunk_id"], "good")


if __name__ == "__main__":
    unittest.main()
