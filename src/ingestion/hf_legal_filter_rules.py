from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_match_text(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d").replace("Đ", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = re.sub(r"[^0-9a-z\s/.-]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _extract_text(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    title = str(
        record.get("doc_title")
        or record.get("title")
        or record.get("name")
        or record.get("document_title")
        or ""
    ).strip()
    content = str(
        record.get("content")
        or record.get("text")
        or record.get("full_text")
        or record.get("content_html")
        or ""
    ).strip()
    doc_type = str(record.get("doc_type") or record.get("loai_van_ban") or "").strip()
    issuer = str(record.get("issuer") or record.get("co_quan_ban_hanh") or "").strip()
    doc_number = str(record.get("doc_number") or record.get("so_ky_hieu") or "").strip()
    return title, content, doc_type, issuer, doc_number


RULE_GROUPS = [
    {
        "matched_group": "business_sme",
        "priority": 1,
        "reason": "matched SME/business support legal phrases",
        "target_domains": ["sme_support", "business_law"],
        "exact_keywords": [
            "luat doanh nghiep",
            "luat ho tro doanh nghiep nho va vua",
            "doanh nghiep nho va vua",
            "ho tro doanh nghiep nho va vua",
            "nghi dinh 80/2021/nd-cp",
            "nghi dinh 01/2021/nd-cp",
            "nghi dinh 47/2021/nd-cp",
            "nghi dinh 122/2021/nd-cp",
            "nghi dinh 39/2018/nd-cp",
        ],
        "weak_keywords": [
            "dnnvv",
            "sme",
            "dang ky doanh nghiep",
            "thanh lap doanh nghiep",
            "giai the doanh nghiep",
            "pha san doanh nghiep",
            "ho kinh doanh",
            "quy phat trien doanh nghiep nho va vua",
            "quy bao lanh tin dung",
            "co so uom tao",
            "khu lam viec chung",
            "khoi nghiep sang tao",
            "chuoi gia tri",
            "cum lien ket nganh",
        ],
    },
    {
        "matched_group": "tax_invoice_accounting",
        "priority": 1,
        "reason": "matched tax/invoice/accounting legal phrases",
        "target_domains": ["tax", "invoice", "accounting"],
        "exact_keywords": [
            "luat quan ly thue",
            "nghi dinh 123/2020/nd-cp",
            "nghi dinh 70/2025/nd-cp",
            "thong tu 78/2021/tt-btc",
            "luat ke toan",
            "thong tu 132/2018/tt-btc",
            "thong tu 133/2016/tt-btc",
            "hoa don dien tu",
            "hoa don co ma cua co quan thue",
        ],
        "weak_keywords": [
            "ma so thue",
            "khai thue",
            "no thue",
            "cuong che no thue",
            "ngung su dung hoa don",
            "bao cao tai chinh",
            "so ke toan",
            "chung tu ke toan",
        ],
    },
    {
        "matched_group": "labor_bhxh_union",
        "priority": 1,
        "reason": "matched labor/social insurance/union legal phrases",
        "target_domains": ["labor", "social_insurance", "union", "occupational_safety"],
        "exact_keywords": [
            "bo luat lao dong",
            "bo luat lao dong 2019",
            "nghi dinh 12/2022/nd-cp",
            "luat bao hiem xa hoi",
            "bao hiem xa hoi",
            "bhxh",
            "luat viec lam",
            "luat an toan ve sinh lao dong",
            "luat cong doan",
            "hop dong lao dong",
            "cham dong bao hiem xa hoi",
        ],
        "weak_keywords": [
            "nguoi lao dong",
            "nguoi su dung lao dong",
            "tai nan lao dong",
            "thu viec",
            "sa thai",
        ],
    },
    {
        "matched_group": "intellectual_property",
        "priority": 1,
        "reason": "matched intellectual property legal phrases",
        "target_domains": ["intellectual_property", "copyright", "industrial_property"],
        "exact_keywords": [
            "luat so huu tri tue",
            "nghi dinh 65/2023/nd-cp",
            "nghi dinh 17/2023/nd-cp",
            "so huu tri tue",
            "quyen tac gia",
            "quyen lien quan",
            "nhan hieu",
            "kieu dang cong nghiep",
            "chi dan dia ly",
            "ten thuong mai",
            "bi mat kinh doanh",
        ],
        "weak_keywords": [
            "shtt",
        ],
    },
    {
        "matched_group": "commerce_procurement_customs_logistics",
        "priority": 1,
        "reason": "matched commerce/procurement/customs/logistics legal phrases",
        "target_domains": [
            "commerce",
            "procurement",
            "customs",
            "logistics",
            "transportation",
            "ecommerce",
            "consumer_protection",
        ],
        "exact_keywords": [
            "luat thuong mai",
            "luat dau thau",
            "dau thau",
            "nha thau",
            "ho so du thau",
            "nhuong quyen thuong mai",
            "dich vu logistics",
            "thuong mai dien tu",
            "bao ve nguoi tieu dung",
        ],
        "weak_keywords": [
            "khuyen mai",
            "logistics",
            "xuat nhap khau",
            "hai quan",
        ],
    },
]

EXCLUDE_KEYWORDS = [
    "bo nhiem",
    "dieu dong",
    "thanh lap ban chi dao",
    "ke hoach to chuc",
    "lich hop",
    "noi dung hop",
    "phan bo ngan sach",
    "du toan ngan sach",
    "thong bao ket luan",
    "giao chi tieu ke hoach",
]

NORMATIVE_TYPES = {
    "luat",
    "bo luat",
    "nghi dinh",
    "thong tu",
    "thong tu lien tich",
    "nghi quyet",
    "quyet dinh",
    "chi thi",
}

RESTRICTED_LOCAL_ISSUERS = [
    "ubnd tinh",
    "ubnd huyen",
    "ubnd thanh pho",
    "so ",
    "chi cuc ",
    "phong ",
]

STRONG_PUBLIC_ISSUERS = [
    "quoc hoi",
    "uy ban thuong vu quoc hoi",
    "chinh phu",
    "thu tuong chinh phu",
    "bo tai chinh",
    "bo lao dong",
    "bo lao dong thuong binh va xa hoi",
    "bo ke hoach va dau tu",
    "bo cong thuong",
    "bo khoa hoc va cong nghe",
    "bo y te",
    "tong cuc thue",
    "bao hiem xa hoi viet nam",
]


def _is_noise_document(title_norm: str, content_norm: str) -> bool:
    if not any(keyword in title_norm or keyword in content_norm for keyword in EXCLUDE_KEYWORDS):
        return False
    core = [
        "doanh nghiep nho va vua",
        "luat doanh nghiep",
        "hoa don dien tu",
        "quan ly thue",
        "bao hiem xa hoi",
        "bo luat lao dong",
        "so huu tri tue",
        "luat dau thau",
        "hai quan",
    ]
    return not any(item in f"{title_norm}\n{content_norm}" for item in core)


def _is_normative_type(doc_type_norm: str) -> bool:
    return any(doc_type_norm.startswith(item) for item in NORMATIVE_TYPES)


def _is_local_issuer_noise(issuer_norm: str) -> bool:
    return any(token in issuer_norm for token in RESTRICTED_LOCAL_ISSUERS)


def _is_strong_public_issuer(issuer_norm: str) -> bool:
    return any(token in issuer_norm for token in STRONG_PUBLIC_ISSUERS)


def _group_matches(group: dict[str, Any], title_norm: str, content_norm: str) -> tuple[list[str], list[str]]:
    exact_title_matches = [keyword for keyword in group["exact_keywords"] if keyword in title_norm]
    exact_content_matches = [keyword for keyword in group["exact_keywords"] if keyword in content_norm]
    weak_matches = [
        keyword for keyword in group["weak_keywords"] if keyword in title_norm or keyword in content_norm
    ]
    return list(dict.fromkeys(exact_title_matches)), list(dict.fromkeys(exact_content_matches)), weak_matches


def evaluate_hf_legal_filter(record: dict[str, Any]) -> dict[str, Any]:
    title, content, doc_type, issuer, _doc_number = _extract_text(record)
    title_norm = normalize_match_text(title)
    content_norm = normalize_match_text(content)
    doc_type_norm = normalize_match_text(doc_type)
    issuer_norm = normalize_match_text(issuer)

    if _is_noise_document(title_norm, content_norm):
        return {
            "include": False,
            "matched_group": "excluded_noise",
            "domain": "",
            "candidate_domains": [],
            "matched_keywords": [],
            "priority": 999,
            "reason": "matched noisy administrative/internal document keywords",
        }

    approved_context = _is_normative_type(doc_type_norm)
    local_noise = _is_local_issuer_noise(issuer_norm)
    public_issuer = _is_strong_public_issuer(issuer_norm)

    matched_keywords: list[str] = []
    candidate_domains: list[str] = []
    matched_group = ""
    priority = 99
    reason = "no target legal domain matched"

    for group in RULE_GROUPS:
        exact_title_matches, exact_content_matches, weak_matches = _group_matches(group, title_norm, content_norm)
        if not exact_title_matches and not exact_content_matches and not weak_matches:
            continue

        include_group = False
        exact_matches: list[str] = []
        if exact_title_matches:
            include_group = True
            exact_matches.extend(exact_title_matches)
        elif exact_content_matches and approved_context and public_issuer and not local_noise:
            include_group = True
            exact_matches.extend(exact_content_matches)
        elif weak_matches and approved_context and not local_noise:
            if any(keyword in title_norm for keyword in weak_matches):
                include_group = True
            elif len(weak_matches) >= 2:
                include_group = True
            elif public_issuer and len(weak_matches) >= 1:
                include_group = True

        if not include_group:
            continue

        if not matched_group or group["priority"] < priority:
            matched_group = str(group["matched_group"])
            priority = int(group["priority"])
            reason = str(group["reason"])
        matched_keywords.extend(exact_matches)
        matched_keywords.extend(weak_matches)
        candidate_domains.extend(group["target_domains"])

    candidate_domains = list(dict.fromkeys(candidate_domains))
    matched_keywords = list(dict.fromkeys(matched_keywords))
    include = bool(candidate_domains)
    domain = candidate_domains[0] if candidate_domains else ""

    return {
        "include": include,
        "matched_group": matched_group,
        "domain": domain,
        "candidate_domains": candidate_domains,
        "matched_keywords": matched_keywords,
        "priority": priority if include else 999,
        "reason": reason,
    }
