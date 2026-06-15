from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List


TAXONOMY_PATH = Path("data/sources/domain_taxonomy.json")

SIMPLE_VECTOR = "SIMPLE_VECTOR"
PARENT_CONTEXT = "PARENT_CONTEXT"
LEGAL_GRAPH_CONTEXT = "LEGAL_GRAPH_CONTEXT"
CROSS_DOMAIN_CONTEXT = "CROSS_DOMAIN_CONTEXT"
MULTI_DOMAIN_COMPLEX = "MULTI_DOMAIN_COMPLEX"

MANUAL_DOMAIN_RULES = {
    "investment_law": [
        "nha dau tu nuoc ngoai",
        "nguoi nuoc ngoai",
        "fdi",
        "irc",
        "giay chung nhan dang ky dau tu",
        "gop von cua nha dau tu nuoc ngoai",
    ],
    "tax_law": [
        "thue",
        "thuế",
        "hoa don",
        "hoa đơn",
        "hoa don dien tu",
        "chung tu",
        "chứng từ",
        "ma so thue",
        "mã số thuế",
        "co quan thue",
        "cơ quan thuế",
        "khau tru thue",
        "khấu trừ thuế",
        "dang ky thue",
        "đăng ký thuế",
        "ngung su dung hoa don",
        "thue gtgt",
        "thuế gtgt",
        "thue tncn",
        "thuế tncn",
        "thue tndn",
        "thuế tndn",
        "thue xuat nhap khau",
        "thue nhà đất",
        "thue mon bai",
        "thuế môn bài",
        "khai thue",
        "khai thuế",
        "nop thue",
        "nộp thuế",
    ],
    "accounting_law": [
        "ke toan",
        "kế toán",
        "so sach ke toan",
        "sổ sách kế toán",
        "bao cao tai chinh",
        "báo cáo tài chính",
        "chung tu ke toan",
        "hach toan",
        "hạch toán",
        "du phong",
        "dự phòng",
        "quyet toan",
        "quyết toán",
        "kiem toan",
        "kiểm toán",
    ],
    "labor_law": [
        "nguoi lao dong",
        "người lao động",
        "nhan vien",
        "nhân viên",
        "hop dong lao dong",
        "hợp đồng lao động",
        "thu viec",
        "thử việc",
        "sa thai",
        "sa thải",
        "lam them gio",
        "làm thêm giờ",
        "luong",
        "lương",
        "giu ban chinh",
        "giữ bản chính",
        "an toan ve sinh lao dong",
        "an toàn vệ sinh lao động",
        "nghi viec",
        "nghỉ việc",
        "nghi phep",
        "nghỉ phép",
        "bao hiem lao dong",
        "bảo hiểm lao động",
        "tien luong",
        "tiền lương",
        "phu cap",
        "phụ cấp",
        "thoi gio lam viec",
        "thời giờ làm việc",
    ],
    "social_insurance": [
        "bhxh",
        "bảo hiểm xã hội",
        "bao hiem xa hoi",
        "bhyt",
        "bảo hiểm y tế",
        "bao hiem y te",
        "bhtn",
        "bảo hiểm thất nghiệp",
        "bao hiem that nghiep",
        "bao hiem tu nguyen",
        "bảo hiểm tự nguyện",
        "dong bao hiem",
        "đóng bảo hiểm",
        "cham dong bao hiem",
        "chậm đóng bảo hiểm",
        "truy thu bao hiem",
        "muc dong bao hiem",
        "mức đóng bảo hiểm",
        "tien luong dong bao hiem",
        "tiền lương đóng bảo hiểm",
    ],
    "administrative_penalty": [
        "xu phat",
        "xử phạt",
        "bi phat",
        "bị phạt",
        "muc phat",
        "mức phạt",
        "vi pham hanh chinh",
        "vi phạm hành chính",
        "khac phuc hau qua",
        "khắc phục hậu quả",
        "che tai",
        "chế tài",
        "phat tien",
        "phạt tiền",
        "phat canh cao",
        "phạt cảnh cáo",
        "tich thu",
        "tịch thu",
    ],
    "civil_commercial_law": [
        "hop dong",
        "hợp đồng",
        "thuong mai",
        "thương mại",
        "nhuong quyen",
        "nhượng quyền",
        "ten thuong mai",
        "tên thương mại",
        "dai ly thuong mai",
        "đại lý thương mại",
        "logistics",
        "boi thuong",
        "bồi thường",
        "tranh chap",
        "tranh chấp",
    ],
    "ip_law": [
        "so huu tri tue",
        "sở hữu trí tuệ",
        "so huu cong nghiep",
        "sở hữu công nghiệp",
        "nhan hieu",
        "nhãn hiệu",
        "sang che",
        "sáng chế",
        "kieu dang cong nghiep",
        "kiểu dáng công nghiệp",
        "quyen tac gia",
        "quyền tác giả",
        "ten thuong mai",
        "chi dan dia ly",
        "chỉ dẫn địa lý",
        "van bang bao ho",
        "văn bằng bảo hộ",
        "tham dinh noi dung",
        "thẩm định nội dung",
    ],
}

FAMILY_MAP = {
    "tax_law": "tax_accounting",
    "accounting_law": "tax_accounting",
    "labor_law": "labor_social",
    "social_insurance": "labor_social",
    "ip_law": "ip",
    "civil_commercial_law": "commercial",
    "investment_law": "investment",
    "administrative_penalty": "penalty",
}


def _load_taxonomy() -> Dict[str, dict]:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    lowered = (
        (text or "")
        .lower()
        .replace("đ", "d")
        .replace("Đ", "d")
        .replace("Ä‘", "d")
        .replace("Ä", "d")
        .replace("Ã„â€˜", "d")
        .replace("Ã„Â", "d")
    )
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _domain_families(domains: Iterable[str]) -> set[str]:
    families = set()
    for domain in domains:
        family = FAMILY_MAP.get(str(domain))
        if family:
            families.add(family)
    return families


# Additional keywords for enterprise/business law to strengthen detection
ENTERPRISE_KEYWORDS = [
    "doanh nghiep",
    "doanh nghiệp",
    "sme",
    "dnvv",
    "doanh nghiep nho",
    "doanh nghiệp nhỏ",
    "doanh nghiep vua",
    "doanh nghiệp vừa",
    "ho kinh doanh",
    "hộ kinh doanh",
    "cong ty",
    "công ty",
    "thanh lap",
    "thành lập",
    "dang ky",
    "đăng ký",
    "giai the",
    "giải thể",
    "phá sản",
    "phá sản",
    "chuyen doi",
    "chuyển đổi",
    "hop nhat",
    "hợp nhất",
    "sáp nhập",
    "sáp nhập",
    "co dong",
    "cổ đông",
    "thanh vien",
    "thành viên",
    "ban quan ly",
    "ban quản lý",
    "hoi dong",
    "hội đồng",
    "von dieu le",
    "vốn điều lệ",
    "loai hinh",
    "loại hình",
    "cong ty tnhh",
    "công ty tnhh",
    "cong ty co phan",
    "công ty cổ phần",
    "doanh nghiep tu nhan",
    "doanh nghiệp tư nhân",
    "hop danh",
    "hợp danh",
    "uu dai",
    "ưu đãi",
    "ho tro",
    "hỗ trợ",
    "thu tuc",
    "thủ tục",
    "giay phep",
    "giấy phép",
    "dieu kien",
    "điều kiện",
]


def detect_domains(query: str) -> List[str]:
    taxonomy = _load_taxonomy()
    lowered = _normalize(query)
    detected = []
    
    # Check if enterprise-related first
    is_enterprise = any(kw in lowered for kw in ENTERPRISE_KEYWORDS)
    if is_enterprise:
        detected.append("business_law")
    
    for domain, keywords in MANUAL_DOMAIN_RULES.items():
        if any(keyword in lowered for keyword in keywords):
            detected.append(domain)
    for domain, meta in taxonomy.items():
        if domain == "business_law":
            continue
        if any(_normalize(keyword) in lowered for keyword in meta.get("keywords", [])):
            detected.append(domain)
    if "nuoc ngoai" in lowered and ("gop von" in lowered or "dau tu" in lowered):
        detected.append("investment_law")
    
    # Always include business_law if no other domains detected, or if it's enterprise-related
    if not detected or is_enterprise:
        if "business_law" not in detected:
            detected.insert(0, "business_law")
    
    return list(dict.fromkeys(detected))


def route_query(query: str, seed_chunks: Iterable[Dict[str, object]] | None = None) -> Dict[str, object]:
    lowered = _normalize(query)
    domains = detect_domains(query)
    seed_chunks = list(seed_chunks or [])
    if any(chunk.get("metadata", {}).get("domain") for chunk in seed_chunks):
        for chunk in seed_chunks:
            domain = chunk.get("metadata", {}).get("domain")
            if domain and domain not in domains:
                domains.append(str(domain))

    route = SIMPLE_VECTOR
    reason = "Default simple retrieval."
    needs_parent = False
    needs_neighbor = False
    needs_graph = False
    needs_cross_domain = False

    satellite_domains = [domain for domain in domains if domain != "business_law"]
    families = _domain_families(satellite_domains)
    broad_question = any(
        token in lowered
        for token in ["toan bo", "nhung viec gi", "can lam gi", "cac nghia vu", "can nhung van ban nao", "so sanh"]
    )
    relationship_question = any(
        token in lowered
        for token in ["lien quan", "can cu", "theo quy dinh tai", "huong dan boi", "sua doi", "bo sung", "thay the", "het hieu luc", "con hieu luc", "ngoai le", "tru truong hop"]
    )
    article_grounding = any(
        token in lowered
        for token in ["dieu", "khoan", "diem", "doi tuong nao", "truong hop nao", "dieu kien", "quyen", "nghia vu", "ai khong duoc", "xu phat", "bi phat", "muc phat"]
    )
    explicit_multi_domain = any(
        token in lowered
        for token in [
            "dong thoi",
            "ket hop",
            "bao gom ca",
            "lien quan den",
            "vua ... vua",
            "nhuong quyen thuong mai",
            "xuat nhap khau",
        ]
    )

    cross_domain_families = families - {"penalty"}
    single_family_with_penalty = len(cross_domain_families) == 1 and "penalty" in families

    if broad_question and (explicit_multi_domain or len(cross_domain_families) > 1):
        route = MULTI_DOMAIN_COMPLEX
        reason = "Query asks for broad guidance across multiple legal families."
        needs_parent = True
        needs_neighbor = True
        needs_graph = True
        needs_cross_domain = True
    elif explicit_multi_domain or len(cross_domain_families) > 1:
        route = CROSS_DOMAIN_CONTEXT
        reason = "Query clearly spans more than one legal family."
        needs_parent = True
        needs_graph = True
        needs_cross_domain = True
    elif relationship_question:
        route = LEGAL_GRAPH_CONTEXT
        reason = "Query asks for legal relationship or legal status expansion."
        needs_parent = True
        needs_graph = True
    elif article_grounding or satellite_domains or single_family_with_penalty:
        route = PARENT_CONTEXT
        reason = "Query requires article/clause grounding within one legal family."
        needs_parent = True

    return {
        "route": route,
        "domains": domains,
        "needs_parent": needs_parent,
        "needs_neighbor": needs_neighbor,
        "needs_graph": needs_graph,
        "needs_cross_domain": needs_cross_domain,
        "reason": reason,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Route legal query to retrieval strategy.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    print(json.dumps(route_query(args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
