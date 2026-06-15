from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple, List


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
    {
        "matched_group": "land_law",
        "priority": 1,
        "reason": "matched land use/lease legal phrases",
        "target_domains": ["land_law", "land_use", "land_lease"],
        "exact_keywords": [
            "luat dat dai",
            "nghi dinh 43/2014/nd-cp",
            "nghi dinh 45/2014/nd-cp",
            "thue dat",
            "su dung dat",
            "quyen su dung dat",
            "giay chung nhan quyen su dung dat",
            "giao dat",
            "cho thue dat",
            "thu hoi dat",
        ],
        "weak_keywords": [
            "dat dai",
            "mat bang",
            "quy hoach dat",
            "dat san xuat",
            "dat kinh doanh",
            "tien thue dat",
        ],
    },
    {
        "matched_group": "administrative_penalty",
        "priority": 1,
        "reason": "matched administrative penalty legal phrases",
        "target_domains": ["administrative_penalty", "penalty", "fine"],
        "exact_keywords": [
            "xuc phat vi pham hanh chinh",
            "nghi dinh 12/2022/nd-cp",
            "nghi dinh 125/2020/nd-cp",
            "muc phat",
            "khung phat",
            "bi phat",
            "phat tien",
            "phat hanh chinh",
        ],
        "weak_keywords": [
            "xuc phat",
            "vi pham",
            "hanh chinh",
            "phat",
            "tien phat",
            "muc phat tien",
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
        "luat dat dai",
        "thue dat",
        "su dung dat",
        "xuc phat vi pham hanh chinh",
        "muc phat",
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


# ──────────────────────────────────────────────
# Date parsing & time-based filtering helpers
# ──────────────────────────────────────────────

_VI_DATE_PATTERNS: List[Tuple[re.Pattern, bool]] = [
    # "ngày 01 tháng 01 năm 2020"
    (re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE), False),
    # "01/01/2020" or "1/1/2020"
    (re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})"), False),
    # ISO "2020-01-01"
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), True),
]


def _parse_vi_date(raw: Any) -> Optional[date]:
    """Robustly parse a Vietnamese date string into a Python date."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    for pat, iso_order in _VI_DATE_PATTERNS:
        m = pat.search(s)
        if m:
            if iso_order:
                y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else:
                d, mon, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return date(y, mon, d)
            except ValueError:
                continue
    return None


def extract_legal_dates(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Extract issued_date, effective_date, expiry_date, status from a HF document dict."""
    meta = doc.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}

    issued: Optional[date] = None
    effective: Optional[date] = None
    expiry: Optional[date] = None
    status: Optional[str] = None

    # 1. issued_date (ngày ban hành) — separate from effective_date
    for iss_key in ("date_issued", "ngay_ban_hanh", "issued_date"):
        val = meta.get(iss_key) or doc.get(iss_key)
        if val:
            issued = _parse_vi_date(val)
            if issued:
                break

    # 2. effective_date — ONLY from explicit fields or "có hiệu lực từ ngày" regex
    for eff_key in ("effective_date", "ngay_co_hieu_luc", "hieu_luc_tu"):
        val = meta.get(eff_key) or doc.get(eff_key)
        if val:
            effective = _parse_vi_date(val)
            if effective:
                break

    # 3. expiry_date
    for exp_key in ("expiry_date", "ngay_het_hieu_luc", "het_hieu_luc", "expiration"):
        val = meta.get(exp_key) or doc.get(exp_key)
        if val:
            expiry = _parse_vi_date(val)
            if expiry:
                break

    # 4. status from metadata
    for st_key in ("status", "trang_thai", "tinh_trang", "hieu_luc"):
        val = meta.get(st_key) or doc.get(st_key)
        if val:
            status = str(val).strip()
            break

    # ── Regex fallback (only on first 5000 chars to avoid matching random dates inside content) ──
    title = str(doc.get("title") or "").strip()
    content = str(doc.get("content") or "").strip()
    # Use preamble only (first part of document) for date inference
    preamble = (title + "\n" + content[:5000]).strip()

    if effective is None:
        # Match: "có hiệu lực từ ngày dd/mm/yyyy" or "hiệu lực thi hành từ ngày ..."
        m = re.search(
            r"(?:có\s+hiệu\s+lực|hiệu\s+lực\s+thi\s+hành)\s+từ\s+ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
            preamble, re.IGNORECASE,
        )
        if m:
            effective = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        else:
            m2 = re.search(
                r"(?:có\s+hiệu\s+lực|hiệu\s+lực\s+thi\s+hành)\s+từ\s+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
                preamble, re.IGNORECASE,
            )
            if m2:
                effective = date(int(m2.group(3)), int(m2.group(2)), int(m2.group(1)))

    if expiry is None:
        m = re.search(
            r"hết\s+hiệu\s+lực\s+(?:vào\s+)?ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
            preamble, re.IGNORECASE,
        )
        if m:
            expiry = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        else:
            m2 = re.search(
                r"ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4}).*?hết\s+hiệu\s+lực",
                preamble, re.IGNORECASE,
            )
            if m2:
                expiry = date(int(m2.group(3)), int(m2.group(2)), int(m2.group(1)))

    if status is None:
        lower = preamble.lower()
        if "hết hiệu lực" in lower or "bị thay thế" in lower or "đã bị bãi bỏ" in lower:
            status = "Hết hiệu lực"
        elif "còn hiệu lực" in lower or "đang có hiệu lực" in lower:
            status = "Vẫn còn hiệu lực"
        else:
            status = "Vẫn còn hiệu lực"  # default assumption

    return {
        "issued_date": issued.isoformat() if issued else None,
        "effective_date": effective.isoformat() if effective else None,
        "expiry_date": expiry.isoformat() if expiry else None,
        "status": status,
    }


def is_valid_on_date(doc_dates: Dict[str, Any], cutoff: date) -> bool:
    """Return True iff effective_date <= cutoff AND still valid on cutoff."""
    eff_str = doc_dates.get("effective_date")
    exp_str = doc_dates.get("expiry_date")
    status = (doc_dates.get("status") or "").strip().lower()

    if eff_str:
        try:
            effective = date.fromisoformat(eff_str)
            if effective > cutoff:
                return False
        except ValueError:
            pass

    if exp_str:
        try:
            expiry = date.fromisoformat(exp_str)
            if expiry <= cutoff:
                return False
        except ValueError:
            pass
    else:
        if status not in {"vẫn còn hiệu lực", "còn hiệu lực", "đang có hiệu lực", "valid", "active"}:
            return False

    return True
