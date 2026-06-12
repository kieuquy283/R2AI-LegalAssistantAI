from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from rag.config.runtime import RetrievalRuntimeConfig, get_retrieval_runtime_config
from rag.retrieval.vectorstore import get_embeddings
from src.retrieval.qdrant_store import QdrantStore

_log = logging.getLogger(__name__)


LEGAL_REF_PATTERN = re.compile(r"\b\d+(?:/\d+)+/[A-Z0-9À-ỴĂÂĐÊÔƠƯ\-]+\b", re.IGNORECASE)

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "labor": [
        "thử việc", "lương", "tiền lương", "người lao động", "hợp đồng lao động",
        "sa thải", "kỷ luật lao động", "bảo hiểm xã hội", "bhxh", "bảo hiểm thất nghiệp",
        "trợ cấp thất nghiệp", "tai nạn lao động", "lao động nữ", "thử viec",
        "luong", "tien luong", "nguoi lao dong", "hop dong lao dong", "sa thai",
        "ky luat lao dong", "bao hiem xa hoi", "bao hiem that nghiep", "tro cap that nghiep",
        "tai nan lao dong", "lao dong nu", "nghỉ việc", "nghi viec",
    ],
    "tax": [
        "thuế", "thue", "quyết toán thuế", "quyet toan thue", "kê khai thuế",
        "ke khai thue", "mã số thuế", "ma so thue", "miễn thuế", "mien thue",
        "giảm thuế", "giam thue", "hoàn thuế", "hoan thue", "thu nhập chịu thuế",
        "thu nhap chiu thue", "thuế suất", "thue suat",
    ],
    "invoice": [
        "hóa đơn", "hoá đơn", "hoa don", "hóa đơn điện tử", "hoa don dien tu",
        "ngừng sử dụng hóa đơn", "ngung su dung hoa don", "hóa đơn đỏ",
    ],
    "accounting": [
        "kế toán", "ke toan", "kiểm toán", "kiem toan", "báo cáo tài chính",
        "bao cao tai chinh", "chứng từ", "chung tu", "sổ sách kế toán",
        "so sach ke toan", "hạch toán", "hach toan",
    ],
    "intellectual_property": [
        "sở hữu trí tuệ", "so huu tri tue", "sở hữu công nghiệp", "so huu cong nghiep",
        "nhãn hiệu", "nhan hieu", "sáng chế", "sang che", "kiểu dáng công nghiệp",
        "kieu dang cong nghiep", "quyền tác giả", "quyen tac gia", "chỉ dẫn địa lý",
        "chi dan dia ly", "văn bằng bảo hộ", "van bang bao ho",
    ],
    "customs": [
        "hải quan", "hai quan", "xuất nhập khẩu", "xuat nhap khau", "thủ tục hải quan",
        "thu tuc hai quan", "thông quan", "thong quan", "hàng hóa xuất khẩu",
        "hang hoa xuat khau", "hàng nhập khẩu", "hang nhap khau",
    ],
    "commerce": [
        "thương mại", "thuong mai", "mua bán hàng hóa", "mua ban hang hoa",
        "cung ứng dịch vụ", "cung ung dich vu", "nhượng quyền", "nhuong quyền",
        "trọng tài thương mại", "trong tai thuong mai", "logistics",
    ],
    "enterprise": [
        "doanh nghiệp", "doanh nghiep", "thành lập doanh nghiệp", "thanh lap doanh nghiep",
        "công ty", "cong ty", "vốn điều lệ", "von dieu le", "góp vốn", "gop von",
        "cổ phần", "co phan", "cổ đông", "co dong", "hội đồng quản trị",
        "hoi dong quan tri", "giám đốc", "giam doc", "đăng ký kinh doanh",
        "dang ky kinh doanh",
    ],
}

_DOMAIN_SYNONYMS: dict[str, list[str]] = {
    "labor": ["labor", "employment", "social_insurance", "labor_law", "social insurance"],
    "tax": ["tax", "tax_law"],
    "invoice": ["invoice", "tax"],
    "accounting": ["accounting", "accounting_law"],
    "intellectual_property": ["intellectual_property", "ip_law", "ip"],
    "customs": ["customs", "custom"],
    "commerce": ["commerce", "commercial", "civil_commercial_law", "trade"],
    "enterprise": ["enterprise", "enterprise_law", "business_law", "corporate"],
}


def _normalize_query_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = lowered.replace("đ", "d").replace("Đ", "d")
    return lowered


def detect_query_domain(query: str) -> str | None:
    normalized = _normalize_query_text(query)
    matched: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in normalized)
        if count > 0:
            matched[domain] = count
    if not matched:
        return None
    return max(matched, key=matched.get)


def analyze_query_domains(query: str) -> dict:
    normalized = _normalize_query_text(query)
    domain_keyword_counts: dict[str, int] = {}
    domain_matched_keywords: dict[str, list[str]] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in normalized]
        if hits:
            domain_keyword_counts[domain] = len(hits)
            domain_matched_keywords[domain] = hits
    if not domain_keyword_counts:
        return {
            "detected_domains": [],
            "primary_domain": None,
            "confidence": None,
            "matched_keywords": {},
            "is_multi_domain": False,
        }
    primary = max(domain_keyword_counts, key=domain_keyword_counts.get)
    max_count = domain_keyword_counts[primary]
    multi = len(domain_keyword_counts) > 1
    if max_count >= 3:
        conf = "high"
    elif max_count >= 1:
        conf = "medium"
    else:
        conf = "low"
    return {
        "detected_domains": list(domain_keyword_counts.keys()),
        "primary_domain": primary,
        "confidence": conf,
        "matched_keywords": domain_matched_keywords,
        "is_multi_domain": multi,
    }


_LABOR_HIGH_KEYWORDS = [
    "thử việc", "thử viec", "hợp đồng lao động", "hop dong lao dong",
    "người lao động", "nguoi lao dong", "tiền lương", "tien luong",
    "sa thải", "sa thai", "bhxh", "bảo hiểm xã hội", "bao hiem xa hoi",
]
_LABOR_MEDIUM_KEYWORDS = [
    "lương", "luong", "nghỉ việc", "nghi viec", "trợ cấp", "tro cap",
    "tai nạn lao động", "tai nan lao dong",
]


def detect_labor_with_confidence(query: str) -> dict:
    normalized = _normalize_query_text(query)
    high_matches = [kw for kw in _LABOR_HIGH_KEYWORDS if kw in normalized]
    medium_matches = [kw for kw in _LABOR_MEDIUM_KEYWORDS if kw in normalized]
    all_keywords = _DOMAIN_KEYWORDS.get("labor", [])
    any_labor_match = (
        any(kw in normalized for kw in all_keywords)
        or bool(high_matches)
        or bool(medium_matches)
    )
    if not any_labor_match:
        return {"detected_domain": None, "domain_confidence": None, "matched_keywords": []}
    if high_matches:
        return {
            "detected_domain": "labor",
            "domain_confidence": "high",
            "matched_keywords": high_matches + medium_matches,
        }
    if medium_matches:
        return {
            "detected_domain": "labor",
            "domain_confidence": "medium",
            "matched_keywords": medium_matches,
        }
    return {
        "detected_domain": "labor",
        "domain_confidence": "low",
        "matched_keywords": [],
    }


_TAX_INVOICE_ACCT_KEYWORDS = [
    "thuế", "thue", "hóa đơn", "hoá đơn", "hoa don", "kế toán", "ke toan",
    "kiểm toán", "kiem toan", "mã số thuế", "ma so thue",
]


def _query_has_tax_invoice_accounting_keywords(query: str) -> bool:
    normalized = _normalize_query_text(query)
    return any(kw in normalized for kw in _TAX_INVOICE_ACCT_KEYWORDS)


def _candidate_domain_matches(candidate: dict[str, Any], domain_group: str) -> bool:
    candidate_domain = str(candidate.get("domain") or "").strip().lower()
    synonyms = _DOMAIN_SYNONYMS.get(domain_group, [domain_group])
    return any(syn == candidate_domain for syn in synonyms)


def _candidate_title_has_tax_invoice_acct(candidate: dict[str, Any]) -> bool:
    title = str(candidate.get("doc_title") or "").lower()
    title = title.replace("đ", "d").replace("Đ", "d")
    for kw in ("hóa đơn", "hoá đơn", "hoa don", "kế toán", "ke toan", "kiểm toán", "kiem toan"):
        if kw in title:
            return True
    return False


def _apply_labor_adjustment_conservative(c: dict[str, Any], score_before: float) -> tuple[float, list[str]]:
    reason_parts: list[str] = []
    adjusted = score_before
    if _candidate_domain_matches(c, "labor"):
        adjusted = score_before * 1.15
        reason_parts.append("same-domain(labor) boost x1.15")
    elif _candidate_domain_matches(c, "tax") or _candidate_domain_matches(c, "invoice") or _candidate_domain_matches(c, "accounting"):
        adjusted = score_before * 0.50
        reason_parts.append(f"wrong-domain({c['candidate_domain']}) penalty x0.50")
        if _candidate_title_has_tax_invoice_acct(c):
            adjusted *= 0.50
            reason_parts.append("invoice/acct title penalty x0.50")
    elif _candidate_domain_matches(c, "commerce") or _candidate_domain_matches(c, "customs"):
        adjusted = score_before * 0.90
        reason_parts.append(f"light-penalty({c['candidate_domain']}) x0.90")
    elif _candidate_domain_matches(c, "enterprise"):
        pass
    else:
        pass
    return adjusted, reason_parts


def _apply_labor_adjustment_aggressive(c: dict[str, Any], score_before: float) -> tuple[float, list[str]]:
    reason_parts: list[str] = []
    adjusted = score_before
    if _candidate_domain_matches(c, "labor"):
        adjusted = score_before * 1.35
        reason_parts.append("same-domain(labor) boost x1.35")
    elif _candidate_domain_matches(c, "tax") or _candidate_domain_matches(c, "invoice") or _candidate_domain_matches(c, "accounting"):
        adjusted = score_before * 0.25
        reason_parts.append(f"wrong-domain({c['candidate_domain']}) penalty x0.25")
        if _candidate_title_has_tax_invoice_acct(c):
            adjusted *= 0.10
            reason_parts.append("invoice/acct title penalty x0.10")
    elif _candidate_domain_matches(c, "commerce"):
        adjusted = score_before * 0.90
        reason_parts.append(f"commerce penalty x0.90")
    elif _candidate_domain_matches(c, "customs"):
        adjusted = score_before * 0.80
        reason_parts.append(f"customs penalty x0.80")
    elif _candidate_domain_matches(c, "enterprise"):
        adjusted = score_before * 1.15
        reason_parts.append("enterprise boost x1.15")
    else:
        pass
    return adjusted, reason_parts


_DEFAULT_TRACE_FIELDS: dict[str, Any] = {
    "domain_rerank_enabled": False,
    "domain_rerank_mode": "conservative",
    "detected_query_domain": None,
    "detected_domains": None,
    "primary_domain": None,
    "is_multi_domain": False,
    "domain_confidence": None,
    "matched_domain_keywords": None,
    "candidate_domain": None,
    "score_before_domain_adjustment": 0.0,
    "score_after_domain_adjustment": 0.0,
    "domain_adjustment_reason": None,
}


def _set_default_trace(candidates: list[dict[str, Any]]) -> None:
    for c in candidates:
        for k, v in _DEFAULT_TRACE_FIELDS.items():
            if k not in c:
                c[k] = v


def apply_domain_adjustment(query: str, candidates: list[dict[str, Any]]) -> str | None:
    import os
    domain_analysis = analyze_query_domains(query)
    _set_default_trace(candidates)
    for c in candidates:
        c["detected_domains"] = list(domain_analysis.get("detected_domains") or [])
        c["primary_domain"] = domain_analysis.get("primary_domain")
        c["is_multi_domain"] = domain_analysis.get("is_multi_domain", False)
        c["candidate_domain"] = str(c.get("domain") or "").strip().lower()
        c["score_before_domain_adjustment"] = float(c.get("dense_score") or 0.0)
    enabled = os.getenv("ENABLE_DOMAIN_AWARE_RERANK", "false").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return None
    mode = os.getenv("DOMAIN_RERANK_MODE", "conservative").strip().lower()
    labor_info = detect_labor_with_confidence(query)
    detected_domain = labor_info["detected_domain"]
    domain_confidence = labor_info["domain_confidence"]
    matched_kws = list(labor_info["matched_keywords"])
    for c in candidates:
        c["domain_rerank_enabled"] = True
        c["domain_rerank_mode"] = mode
        c["detected_query_domain"] = detected_domain
        c["domain_confidence"] = domain_confidence
        c["matched_domain_keywords"] = list(matched_kws)
    if detected_domain != "labor":
        return None
    query_has_tax = _query_has_tax_invoice_accounting_keywords(query)
    if query_has_tax:
        return None
    if mode == "aggressive" and domain_confidence != "high":
        return None
    for c in candidates:
        if mode == "aggressive":
            adjusted, reason = _apply_labor_adjustment_aggressive(c, c["score_before_domain_adjustment"])
        else:
            adjusted, reason = _apply_labor_adjustment_conservative(c, c["score_before_domain_adjustment"])
        c["score_after_domain_adjustment"] = round(adjusted, 6)
        c["dense_score"] = round(adjusted, 6)
        c["final_score"] = round(adjusted, 6)
        c["domain_adjustment_reason"] = "; ".join(reason) if reason else "no adjustment"
    return detected_domain


def _normalize_score(value: float) -> float:
    score = float(value or 0.0)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _minmax_normalize_dense_scores(candidates: list[dict]) -> None:
    scores = [float(c.get("raw_dense_score") or 0.0) for c in candidates]
    if not scores:
        return
    min_score = min(scores)
    max_score = max(scores)
    diff = max_score - min_score
    for c in candidates:
        raw = float(c.get("raw_dense_score") or 0.0)
        if diff > 0:
            c["dense_score"] = round((raw - min_score) / diff, 6)
        else:
            c["dense_score"] = 1.0 if raw > 0 else 0.0


class QdrantRetriever:
    def __init__(
        self,
        *,
        store: QdrantStore | None = None,
        config: RetrievalRuntimeConfig | None = None,
        embeddings: Any | None = None,
    ) -> None:
        self.config = config or get_retrieval_runtime_config()
        self.store = store or QdrantStore(config=self.config)
        self.embeddings = embeddings or get_embeddings()
        self._articles_available = self._check_collection_exists(
            self.config.qdrant_collection_articles
        )
        if not self._articles_available:
            _log.warning(
                "Collection '%s' not found. legal_articles will be skipped. "
                "Only legal_chunks and legal_docs will be used for retrieval.",
                self.config.qdrant_collection_articles,
            )

    def _check_collection_exists(self, collection_name: str) -> bool:
        try:
            collections = self.store.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception:
            _log.exception("Failed to list collections, assuming '%s' is unavailable", collection_name)
            return False

    def _allowed_domain(self, payload: dict[str, Any], preferred_domains: Sequence[str] | None) -> bool:
        return True

    def _query_collection(self, collection_name: str, query_vector: list[float], limit: int) -> list[Any]:
        try:
            return list(
                self.store.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
            )
        except Exception:
            result = self.store.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return list(getattr(result, "points", []) or [])

    def _make_candidate(self, *, level: str, hit: Any, query: str) -> dict[str, Any]:
        payload = dict(getattr(hit, "payload", None) or {})
        raw_dense_score = float(getattr(hit, "score", None) or 0.0)
        article = str(payload.get("article") or "").strip()
        candidate_id = str(
            payload.get("chunk_id")
            or payload.get("node_id")
            or payload.get("doc_id")
            or payload.get("id")
            or ""
        ).strip()
        citation = str(payload.get("citation") or payload.get("title") or payload.get("doc_title") or "").strip()
        doc_number = str(payload.get("doc_number") or "").strip()
        doc_title = str(payload.get("doc_title") or payload.get("title") or "").strip()
        content = str(payload.get("content") or payload.get("cleaned_text") or payload.get("embedding_text") or "").strip()
        legal_ref_match = 1.0 if doc_number and LEGAL_REF_PATTERN.search(query) and doc_number in query else 0.0
        return {
            "candidate_id": f"{level}:{candidate_id}",
            "retrieval_level": level,
            "retrieval_method": "dense",
            "retrieval_source": "qdrant",
            "raw_dense_score": raw_dense_score,
            "dense_score": raw_dense_score,
            "score": raw_dense_score,
            "bm25_score": 0.0,
            "exact_score": legal_ref_match,
            "title_overlap": 0.0,
            "lexical_overlap": 0.0,
            "domain_match": 0.0,
            "domain_score": 0.0,
            "citation_match": 1.0 if article and article.lower() in citation.lower() else 0.0,
            "final_score": raw_dense_score,
            "confidence": raw_dense_score,
            "chunk_id": str(payload.get("chunk_id") or candidate_id),
            "doc_id": str(payload.get("doc_id") or "").strip(),
            "article_id": str(payload.get("node_id") or "").strip(),
            "chunk_ref_id": str(payload.get("chunk_id") or "").strip(),
            "doc_number": doc_number,
            "doc_title": doc_title,
            "article": article,
            "clause": str(payload.get("clause") or "").strip(),
            "citation": citation,
            "domain": str(payload.get("domain") or "").strip(),
            "source_url": str(payload.get("source_url") or "").strip(),
            "content": content,
            "source_dataset": str(payload.get("source_dataset") or "local_corpus"),
            "priority": int(payload.get("priority") or 0),
            "metadata": payload,
        }

    def search(self, query: str, *, preferred_domains: Sequence[str] | None = None) -> list[dict[str, Any]]:
        query_vector = list(self.embeddings.embed_query(query))
        specs = [
            ("doc", self.config.qdrant_collection_docs, max(self.config.candidate_k_docs * 2, self.config.candidate_k_docs)),
            ("chunk", self.config.qdrant_collection_chunks, max(self.config.candidate_k_chunks * 2, self.config.candidate_k_chunks)),
        ]
        if self._articles_available:
            specs.insert(
                1,
                (
                    "article",
                    self.config.qdrant_collection_articles,
                    max(self.config.candidate_k_articles * 2, self.config.candidate_k_articles),
                ),
            )
        candidates: list[dict[str, Any]] = []
        for level, collection_name, limit in specs:
            for hit in self._query_collection(collection_name, query_vector, limit):
                payload = dict(getattr(hit, "payload", None) or {})
                if not self._allowed_domain(payload, preferred_domains):
                    continue
                candidates.append(self._make_candidate(level=level, hit=hit, query=query))
        _minmax_normalize_dense_scores(candidates)
        qdrant_path_val = str(self.config.qdrant_path) if self.config.qdrant_path else ""
        qdrant_mode_val = str(getattr(self.store, "mode", "unknown"))
        for c in candidates:
            c["qdrant_path"] = qdrant_path_val
            c["qdrant_mode"] = qdrant_mode_val
            c["articles_available"] = self._articles_available
        return candidates
