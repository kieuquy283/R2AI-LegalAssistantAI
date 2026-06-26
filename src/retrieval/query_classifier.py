"""LLM Query Classifier — Phase 5: classifies query type for adaptive retrieval strategy."""

import os
import re

from src.retrieval.legal_exact_search import LEGAL_REF_PATTERN


_MUC_PHAT_KW = [
    "mức phạt", "xử phạt", "vi phạm", "phạt tiền", "mức xử phạt",
    "hình thức xử phạt", "biện pháp khắc phục", "mức phạt tiền",
    "phạt cảnh cáo", "phạt bổ sung",
]
_THU_TUC_KW = [
    "thủ tục", "đăng ký", "xin cấp", "xin giấy", "trình tự",
    "hồ sơ", "thẩm quyền", "cấp giấy", "gia hạn",
]
_DINH_NGHIA_KW = [
    "là gì", "là ai", "định nghĩa", "khái niệm", "thế nào là",
    "hiểu như thế nào", "được hiểu là", "bao gồm những gì",
]
_SO_SANH_KW = [
    "so sánh", "khác nhau", "phân biệt", "giống và khác",
    "ưu điểm", "nhược điểm",
]


def classify_query(query: str) -> dict:
    """Classify query type using rule-based heuristics.
    
    Returns dict with:
    - types: list of detected types
    - has_legal_ref: whether query contains a legal document reference
    - is_specific: whether query is specific (has ref or penalty type)
    - boost_exact: whether exact search should be prioritized
    """
    lowered = query.lower()
    types = set()
    has_ref = bool(LEGAL_REF_PATTERN.search(query))

    if any(kw in lowered for kw in _MUC_PHAT_KW):
        types.add("muc_phat")
    if any(kw in lowered for kw in _THU_TUC_KW):
        types.add("thu_tuc")
    if any(kw in lowered for kw in _DINH_NGHIA_KW):
        types.add("dinh_nghia")
    if any(kw in lowered for kw in _SO_SANH_KW):
        types.add("so_sanh")
    if has_ref:
        types.add("co_so_hieu")

    return {
        "types": sorted(types),
        "has_legal_ref": has_ref,
        "is_specific": has_ref or "muc_phat" in types or "thu_tuc" in types,
        "boost_exact": has_ref,
    }
