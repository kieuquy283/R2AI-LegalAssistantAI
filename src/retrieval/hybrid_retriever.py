from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import pickle
import os
import unicodedata
from collections import defaultdict
from heapq import nlargest
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import faiss
import numpy as np

from rag.modules.retrieval.utils import tokenize_for_bm25
from rag.retrieval.vectorstore import get_embeddings
from src.ingestion.common import read_jsonl


TOPIC_RULES = [
    {
        "name": "sme_support_innovation",
        "keywords": ["co so uom tao", "khu lam viec chung", "ho tro doanh nghiep nho va vua", "uom tao"],
        "required_phrases": [
            "uom tao",
            "khu lam viec chung",
            "doanh nghiep nho va vua",
            "ho tro doanh nghiep nho va vua",
            "80/2021",
            "34/2018",
            "39/2019",
        ],
        "title_phrases": [
            "uom tao",
            "khu lam viec chung",
            "doanh nghiep nho va vua",
            "ho tro doanh nghiep nho va vua",
        ],
        "preferred_domains": ["business_law", "investment_law"],
        "missing_penalty": 0.55,
        "boost": 0.45,
    },
    {
        "name": "dnnvv_procurement",
        "keywords": ["uu dai dau thau", "dau thau", "doanh nghiep nho va vua", "dnnvv"],
        "required_phrases": ["dau thau", "doanh nghiep nho va vua", "luat dau thau"],
        "title_phrases": ["dau thau", "doanh nghiep nho va vua", "uu dai"],
        "preferred_domains": ["business_law"],
        "missing_penalty": 0.45,
        "boost": 0.4,
    },
    {
        "name": "invoice_tax_detail",
        "keywords": ["hoa don dien tu", "ma co quan thue", "chu ky so", "luat quan ly thue", "nghi dinh 123/2020"],
        "required_phrases": ["hoa don", "quan ly thue", "123/2020", "70/2025"],
        "title_phrases": ["hoa don", "quan ly thue", "123/2020", "70/2025"],
        "preferred_domains": ["business_law"],
        "missing_penalty": 0.5,
        "boost": 0.5,
    },
    {
        "name": "labor_penalty_certificate",
        "keywords": ["van bang", "chung chi", "bang cap", "giu ban chinh", "nguoi lao dong", "hop dong"],
        "required_phrases": ["van bang", "chung chi", "bang cap", "lao dong", "bo luat lao dong", "12/2022"],
        "title_phrases": ["van bang", "chung chi", "bang cap", "lao dong", "bo luat lao dong"],
        "preferred_domains": ["labor_law", "business_law"],
        "missing_penalty": 0.45,
        "boost": 0.38,
    },
    {
        "name": "copyright_registration",
        "keywords": ["quyen tac gia", "ho so dang ky", "ban quyen", "so huu tri tue"],
        "required_phrases": ["quyen tac gia", "ban quyen", "so huu tri tue", "luat so huu tri tue"],
        "title_phrases": ["quyen tac gia", "ban quyen", "so huu tri tue"],
        "preferred_domains": ["business_law", "investment_law"],
        "missing_penalty": 0.5,
        "boost": 0.42,
    },
    {
        "name": "dnnvv",
        "keywords": ["doanh nghiep nho va vua", "dnnvv", "ho tro dnnvv", "ho tro doanh nghiep nho va vua"],
        "required_phrases": [
            "doanh nghiep nho va vua",
            "dnnvv",
            "ho tro doanh nghiep nho va vua",
            "quy bao lanh tin dung doanh nghiep nho va vua",
            "quy phat trien doanh nghiep nho va vua",
        ],
        "title_phrases": [
            "doanh nghiep nho va vua",
            "ho tro doanh nghiep nho va vua",
            "dnnvv",
            "80/2021",
            "39/2019",
            "34/2018",
        ],
        "preferred_domains": ["business_law", "investment_law"],
        "missing_penalty": 0.5,
        "boost": 0.28,
    },
    {
        "name": "tax_invoice",
        "keywords": ["hoa don", "hoa don dien tu", "ma co quan thue", "chu ky so", "ke khai thue"],
        "required_phrases": ["hoa don", "hoa don dien tu", "chung tu", "quan ly thue", "123/2020", "70/2025"],
        "title_phrases": [
            "quan ly thue",
            "hoa don",
            "chung tu",
            "123/2020",
            "70/2025",
        ],
        "preferred_domains": ["business_law"],
        "missing_penalty": 0.4,
        "boost": 0.24,
    },
    {
        "name": "labor_social",
        "keywords": ["bao hiem xa hoi", "bhxh", "nguoi lao dong", "hop dong lao dong", "cham dong"],
        "required_phrases": ["bao hiem xa hoi", "bhxh", "bo luat lao dong", "lao dong", "12/2022"],
        "title_phrases": [
            "bao hiem xa hoi",
            "bo luat lao dong",
            "lao dong",
            "bhxh",
            "12/2022",
        ],
        "preferred_domains": ["labor_law", "business_law"],
        "missing_penalty": 0.45,
        "boost": 0.26,
    },
    {
        "name": "ip",
        "keywords": [
            "so huu tri tue",
            "so huu cong nghiep",
            "quyen tac gia",
            "ban quyen",
            "nhan hieu",
            "kieu dang cong nghiep",
            "van bang bao ho",
            "tham dinh noi dung",
            "chuyen nhuong quyen so huu cong nghiep",
        ],
        "required_phrases": [
            "so huu tri tue",
            "so huu cong nghiep",
            "quyen tac gia",
            "ban quyen",
            "nhan hieu",
            "van bang bao ho",
            "tham dinh noi dung",
            "chuyen nhuong quyen so huu cong nghiep",
            "65/2023",
            "23/2023",
        ],
        "title_phrases": [
            "so huu tri tue",
            "so huu cong nghiep",
            "quyen tac gia",
            "ban quyen",
            "nhan hieu",
            "van bang bao ho",
            "tham dinh noi dung",
            "chuyen nhuong quyen so huu cong nghiep",
            "65/2023",
            "23/2023",
        ],
        "preferred_domains": ["business_law", "investment_law"],
        "missing_penalty": 0.4,
        "boost": 0.26,
    },
    {
        "name": "procurement_trade",
        "keywords": ["dau thau", "thuong mai", "mua sam", "canh tranh"],
        "required_phrases": ["dau thau", "thuong mai", "mua sam", "175/2024"],
        "title_phrases": [
            "dau thau",
            "thuong mai",
            "mua sam",
            "175/2024",
        ],
        "preferred_domains": ["business_law", "investment_law", "civil_commercial_law"],
        "missing_penalty": 0.3,
        "boost": 0.20,
    },
    {
        "name": "customs_logistics",
        "keywords": ["hai quan", "logistics", "thong quan", "xuat khau", "nhap khau"],
        "required_phrases": ["hai quan", "logistics", "xuat khau", "nhap khau", "thong quan"],
        "title_phrases": [
            "hai quan",
            "logistics",
            "xuat khau",
            "nhap khau",
            "thong quan",
        ],
        "preferred_domains": ["business_law", "investment_law"],
        "missing_penalty": 0.22,
        "boost": 0.16,
    },
]

PHRASE_GUARDS = [
    "giu ban chinh",
    "hoa don dien tu",
    "ma co quan thue",
    "quyen tac gia",
    "uom tao",
    "khu lam viec chung",
    "doanh nghiep nho va vua",
    "dau thau",
    "bao hiem xa hoi",
]

GENERIC_TITLE_PENALTIES = {
    "dnnvv": ["thu tuc hanh chinh", "thanh lap doanh nghiep", "ho kinh doanh", "giai the doanh nghiep"],
    "dnnvv_procurement": ["thu tuc hanh chinh", "thanh lap doanh nghiep", "ho kinh doanh", "bao lanh", "vay von"],
    "tax_procedure": ["giai the doanh nghiep", "quan ly tai chinh tam thoi", "dau thau", "quan ly nha nuoc ve hai quan"],
    "tax_invoice": ["hoan thue", "giai the doanh nghiep", "quan ly tai chinh tam thoi", "thue tncn", "thue gtgt"],
    "invoice_signature": ["hoan thue", "thue tncn", "thue gtgt", "xuat khau", "nhap khau"],
    "labor_social": ["muc luong toi thieu", "thoi gio lam viec", "nghi ngoi doi voi lao dong thoi vu", "nguoi lao dong viet nam di lam viec o nuoc ngoai"],
    "social_insurance_penalty": ["hop dong lao dong", "muc luong toi thieu", "thoi gio lam viec", "bao hiem that nghiep"],
    "ip": [
        "dau tu von nha nuoc",
        "quy phat trien doanh nghiep nho va vua",
        "tuan thu phap luat hai quan",
        "chung chi kiem toan vien",
        "ke toan vien",
        "muc luong toi thieu",
        "thi hanh an dan su",
        "luat hoa chat",
    ],
    "copyright_registration": ["dau tu von nha nuoc", "quy phat trien doanh nghiep nho va vua", "so huu cong nghiep"],
}

SOURCE_CATEGORY_BOOSTS = {
    "tax_procedure": ["/thue/", "/ke-toan/"],
    "tax_invoice": ["/thue/", "/ke-toan/"],
    "invoice_signature": ["/thue/", "/ke-toan/"],
    "labor_social": ["/lao-dong/", "/bao-hiem/"],
    "social_insurance_penalty": ["/bao-hiem/", "/lao-dong/"],
    "ip": ["/so-huu-tri-tue/", "/cong-nghiep/"],
    "copyright_registration": ["/so-huu-tri-tue/"],
    "procurement_trade": ["/dau-thau/", "/thuong-mai/"],
    "dnnvv_procurement": ["/doanh-nghiep/", "/dau-thau/"],
    "customs_logistics": ["/xuat-nhap-khau/", "/hai-quan/"],
    "dnnvv": ["/doanh-nghiep/"],
}


def _normalize_plain(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d").replace("Đ", "d").replace("Ä‘", "d").replace("Ä", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


@dataclass
class _SearchItem:
    chunk_id: str
    score: float
    retrieval_method: str
    content: str
    embedding_text: str
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "score": float(self.score),
            "retrieval_score": float(self.score),
            "retrieval_method": self.retrieval_method,
            "content": self.content,
            "embedding_text": self.embedding_text,
            "metadata": self.metadata,
        }


class _SimpleBM25:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]]) -> None:
        import math

        self._math = math
        self.corpus_tokens = [list(tokens) for tokens in corpus_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.k1 = 1.5
        self.b = 0.75
        self.doc_freq: Dict[str, int] = {}
        for tokens in self.corpus_tokens:
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
        scores: List[float] = []
        total_docs = max(len(self.corpus_tokens), 1)
        for tokens in self.corpus_tokens:
            token_counts: Dict[str, int] = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            doc_len = len(tokens)
            score = 0.0
            for token in query_tokens:
                freq = token_counts.get(token, 0)
                if freq == 0:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = self._math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
                denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avgdl, 1e-8)))
                score += idf * ((freq * (self.k1 + 1)) / max(denom, 1e-8))
            scores.append(score)
        return scores


class _FastBM25:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]]) -> None:
        import math

        self._math = math
        self.corpus_tokens = [list(tokens) for tokens in corpus_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.k1 = 1.5
        self.b = 0.75
        self.doc_freq: Dict[str, int] = {}
        self.postings: Dict[str, List[tuple[int, int]]] = defaultdict(list)
        for doc_idx, tokens in enumerate(self.corpus_tokens):
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, freq in counts.items():
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
                self.postings[token].append((doc_idx, freq))

    def get_scores(self, query_tokens: Sequence[str]) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        total_docs = max(len(self.corpus_tokens), 1)
        for token in query_tokens:
            postings = self.postings.get(token)
            if not postings:
                continue
            df = self.doc_freq.get(token, 0)
            idf = self._math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
            for doc_idx, freq in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avgdl, 1e-8)))
                scores[doc_idx] += idf * ((freq * (self.k1 + 1)) / max(denom, 1e-8))
        return scores


class HybridRetriever:
    def __init__(
        self,
        *,
        faiss_index_path: str | Path = "data/indexes/faiss.index",
        metadata_path: str | Path = "data/indexes/chunk_metadata.json",
        chunks_path: str | Path = "data/processed/chunks.jsonl",
        bm25_corpus_path: str | Path = "data/indexes/bm25_corpus.json",
        embedding_model=None,
        alpha: float = 0.6,
    ) -> None:
        self.faiss_index_path = Path(faiss_index_path)
        self.metadata_path = Path(metadata_path)
        self.chunks_path = Path(chunks_path)
        self.bm25_corpus_path = Path(bm25_corpus_path)
        self.bm25_index_path = self.bm25_corpus_path.with_name("bm25_index.pkl")
        self.embedding_model = embedding_model or get_embeddings()
        self.alpha = float(alpha)

        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.faiss_index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing chunk metadata: {self.metadata_path}")
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Missing chunks file: {self.chunks_path}")

        self.index = faiss.read_index(str(self.faiss_index_path))
        raw_metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if isinstance(raw_metadata, dict):
            self.metadata_rows = list(raw_metadata.values())
        else:
            self.metadata_rows = list(raw_metadata)
        self.chunks = read_jsonl(self.chunks_path)
        self.chunk_by_id = {str(row["chunk_id"]): row for row in self.chunks}
        self.metadata_by_index = {int(row["index"]): row for row in self.metadata_rows}
        self._init_bm25()
        self.phrase_index: Dict[str, List[str]] = {}

    def _init_bm25(self) -> None:
        if self.bm25_corpus_path.exists():
            corpus_rows = json.loads(self.bm25_corpus_path.read_text(encoding="utf-8"))
            self.bm25_chunk_ids = [str(row["chunk_id"]) for row in corpus_rows]
            tokenized = [list(row.get("tokens") or []) for row in corpus_rows]
            normalized_tokenized = [
                tokenize_for_bm25(_normalize_plain(str(row.get("text") or row.get("embedding_text") or row.get("content") or "")))
                for row in corpus_rows
            ]
        else:
            self.bm25_chunk_ids = [str(row["chunk_id"]) for row in self.chunks]
            tokenized = [tokenize_for_bm25(str(row.get("embedding_text") or row.get("content") or "")) for row in self.chunks]
            normalized_tokenized = [
                tokenize_for_bm25(_normalize_plain(str(row.get("embedding_text") or row.get("content") or "")))
                for row in self.chunks
            ]

        self.bm25 = _FastBM25(tokenized)
        self.bm25_phrase = _FastBM25(normalized_tokenized)

    def _phrase_hits(self, phrase: str, *, limit: int = 10) -> List[str]:
        normalized_phrase = _normalize_plain(phrase)
        if not normalized_phrase:
            return []
        cached = self.phrase_index.get(normalized_phrase)
        if cached is not None:
            return cached
        phrase_tokens = tokenize_for_bm25(_normalize_plain(phrase))
        if not phrase_tokens:
            self.phrase_index[normalized_phrase] = []
            return []

        candidate_docs: set[int] | None = None
        for token in phrase_tokens:
            postings = self.bm25_phrase.postings.get(token)
            if not postings:
                self.phrase_index[normalized_phrase] = []
                return []
            token_docs = {doc_idx for doc_idx, _freq in postings}
            candidate_docs = token_docs if candidate_docs is None else candidate_docs & token_docs
            if not candidate_docs:
                self.phrase_index[normalized_phrase] = []
                return []

        bm25_scores = self.bm25_phrase.get_scores(phrase_tokens)
        ranked_candidates = sorted(candidate_docs, key=lambda doc_idx: bm25_scores.get(doc_idx, 0.0), reverse=True)

        hits: List[str] = []
        for doc_idx in ranked_candidates:
            if doc_idx >= len(self.bm25_chunk_ids):
                continue
            chunk_id = self.bm25_chunk_ids[doc_idx]
            chunk = self.chunk_by_id.get(chunk_id)
            if not chunk:
                continue
            text = _normalize_plain(self._combined_text(chunk))
            if normalized_phrase in text:
                hits.append(chunk_id)
                if len(hits) >= limit:
                    break
        self.phrase_index[normalized_phrase] = hits
        return hits

    def _normalize(self, scores: Sequence[float]) -> List[float]:
        values = [float(score) for score in scores]
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if maximum == minimum:
            return [1.0 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def _allowed_domain(self, metadata: Dict[str, object], domain: str | Sequence[str] | None) -> bool:
        if domain is None:
            return True
        allowed = {domain} if isinstance(domain, str) else set(domain)
        return str(metadata.get("domain")) in allowed

    def _combined_text(self, chunk: Dict[str, object]) -> str:
        parts = [
            chunk.get("doc_title"),
            chunk.get("domain"),
            chunk.get("legal_path"),
            chunk.get("citation"),
            chunk.get("article"),
            chunk.get("clause"),
            chunk.get("content"),
            chunk.get("embedding_text"),
        ]
        values = []
        for part in parts:
            text = str(part or "").strip()
            if text:
                values.append(text)
        return "\n".join(values).strip()

    def _topic_rules(self, query: str) -> List[Dict[str, object]]:
        normalized_query = _normalize_plain(query)
        matches: List[Dict[str, object]] = []
        for rule in TOPIC_RULES:
            if any(keyword in normalized_query for keyword in rule["keywords"]):
                matches.append(rule)
        return matches

    def _topic_profile(self, query: str) -> str | None:
        normalized_query = _normalize_plain(query)
        if "doanh nghiep nho va vua" in normalized_query and "dau thau" in normalized_query:
            return "dnnvv_procurement"
        if "bao hiem xa hoi" in normalized_query and any(token in normalized_query for token in ["cham dong", "xu phat", "bi phat", "che tai"]):
            return "social_insurance_penalty"
        if "thue" in normalized_query and any(token in normalized_query for token in ["dang ky", "khai", "nop", "mien", "giam", "no thue", "cuong che", "nop thua", "khau tru"]):
            return "tax_procedure"
        if "hoa don" in normalized_query and any(token in normalized_query for token in ["chu ky so", "ma co quan thue", "dien tu"]):
            return "invoice_signature"
        if any(token in normalized_query for token in ["quyen tac gia", "ho so dang ky", "dang ky quyen tac gia"]):
            return "copyright_registration"
        if any(token in normalized_query for token in ["doanh nghiep nho va vua", "dnnvv", "uom tao", "khu lam viec chung"]):
            return "dnnvv"
        if any(token in normalized_query for token in ["hoa don", "co quan thue", "ma so thue", "ke toan", "chung tu"]):
            return "tax_invoice"
        if any(token in normalized_query for token in ["so huu tri tue", "so huu cong nghiep", "quyen tac gia", "nhan hieu", "ten thuong mai", "van bang bao ho", "tham dinh noi dung"]):
            return "ip"
        if any(token in normalized_query for token in ["bao hiem xa hoi", "bhxh", "nguoi lao dong", "nhan vien", "giu ban chinh"]):
            return "labor_social"
        if any(token in normalized_query for token in ["dau thau", "thuong mai", "nhuong quyen"]):
            return "procurement_trade"
        if any(token in normalized_query for token in ["hai quan", "logistics", "xuat nhap khau", "thong quan"]):
            return "customs_logistics"
        return None

    def _score_context(self, query: str, chunk: Dict[str, object], base_score: float, *, preferred_domains: Sequence[str] | None = None) -> Dict[str, float]:
        normalized_query = _normalize_plain(query)
        query_tokens = set(tokenize_for_bm25(query))
        combined_text = self._combined_text(chunk)
        normalized_text = _normalize_plain(combined_text)
        title_text = " ".join(
            str(part or "")
            for part in [chunk.get("doc_title"), chunk.get("citation"), chunk.get("legal_path")]
            if str(part or "").strip()
        )
        normalized_title = _normalize_plain(title_text)

        text_tokens = set(tokenize_for_bm25(combined_text))
        title_tokens = set(tokenize_for_bm25(title_text))
        query_size = max(len(query_tokens), 1)
        topic_profile = self._topic_profile(query)
        source_url = _normalize_plain(str(chunk.get("source_url") or ""))

        lexical_overlap = len(query_tokens & text_tokens) / query_size if query_tokens else 0.0
        title_match = len(query_tokens & title_tokens) / query_size if query_tokens else 0.0
        citation_match = 1.0 if normalized_query and str(chunk.get("citation") or "").strip() and any(
            token in normalized_title for token in set(tokenize_for_bm25(str(chunk.get("citation") or "")))
        ) else 0.0

        domain_value = str(chunk.get("domain") or "")
        preferred = set(str(domain) for domain in (preferred_domains or []) if str(domain))
        domain_match = 0.0
        if domain_value and domain_value in preferred and domain_value != "business_law":
            domain_match = 1.0
        elif domain_value == "business_law" and preferred == {"business_law"}:
            domain_match = 0.25

        topic_boost = 0.0
        matched_rules = self._topic_rules(query)
        for rule in matched_rules:
            required_hits = [phrase for phrase in rule["required_phrases"] if phrase in normalized_text]
            title_phrases = [phrase for phrase in rule["title_phrases"] if phrase in normalized_title or phrase in normalized_text]
            keyword_hits = [keyword for keyword in rule["keywords"] if keyword in normalized_text]
            phrase_hits = len(required_hits) + len(title_phrases)
            if phrase_hits:
                topic_boost += float(rule["boost"]) * min(3.0, 1.0 * len(required_hits) + 0.6 * len(title_phrases))
                title_match += min(0.45, 0.12 * len(title_phrases))
                lexical_overlap += min(0.15, 0.03 * len(required_hits))
            elif keyword_hits:
                topic_boost += float(rule["boost"]) * 0.2
            else:
                topic_boost -= float(rule.get("missing_penalty", 0.0))
            if keyword_hits:
                topic_boost += min(0.12, 0.03 * len(keyword_hits))
            if required_hits:
                topic_boost += min(0.18, 0.04 * len(required_hits))
            if not required_hits and not title_phrases:
                topic_boost -= max(0.75, float(rule.get("missing_penalty", 0.0)))

        if topic_profile:
            source_hints = SOURCE_CATEGORY_BOOSTS.get(topic_profile, [])
            if source_hints and any(token in source_url for token in source_hints):
                topic_boost += 0.18
            generic_penalties = GENERIC_TITLE_PENALTIES.get(topic_profile, [])
            generic_hits = [token for token in generic_penalties if token in normalized_title]
            if generic_hits:
                topic_boost -= 0.85 + 0.15 * len(generic_hits)

        if topic_profile == "tax_procedure" and all(token not in normalized_text for token in ["thue", "dang ky thue", "khai thue", "nop thue", "mien thue", "giam thue", "xoa tien thue no", "thuong mai dien tu"]):
            topic_boost -= 1.0
        if topic_profile == "tax_invoice" and "hoa don" not in normalized_text and "quan ly thue" not in normalized_text:
            topic_boost -= 0.9
        if topic_profile == "invoice_signature" and all(token not in normalized_text for token in ["hoa don", "dien tu", "chu ky so", "ma co quan thue", "119/2018", "68/2019", "123/2020"]):
            topic_boost -= 1.15
        if topic_profile == "labor_social" and all(token not in normalized_text for token in ["lao dong", "bao hiem xa hoi", "bo luat lao dong", "12/2022"]):
            topic_boost -= 0.85
        if topic_profile == "social_insurance_penalty" and all(token not in normalized_text for token in ["bao hiem xa hoi", "bhxh", "xu phat", "cham dong", "lao dong", "216", "38/2022"]):
            topic_boost -= 1.1
        if topic_profile == "ip" and all(token not in normalized_text for token in ["so huu tri tue", "quyen tac gia", "nhan hieu", "ten thuong mai", "65/2023"]):
            topic_boost -= 0.95
        if topic_profile == "copyright_registration" and all(token not in normalized_text for token in ["quyen tac gia", "ho so dang ky", "giay cam doan", "to khai", "ban sao tac pham", "50/2005"]):
            topic_boost -= 1.05
        if topic_profile == "dnnvv" and all(token not in normalized_text for token in ["doanh nghiep nho va vua", "uom tao", "khu lam viec chung", "80/2021", "39/2019"]):
            topic_boost -= 1.0
        if topic_profile == "dnnvv_procurement" and all(token not in normalized_text for token in ["doanh nghiep nho va vua", "dau thau", "80/2021", "uu dai"]):
            topic_boost -= 1.1

        if topic_profile == "dnnvv_procurement" and "thu tuc hanh chinh" in normalized_title:
            topic_boost -= 1.2
        if topic_profile == "tax_procedure" and all(token not in normalized_title for token in ["thue", "quan ly thue", "dang ky thue", "khai thue", "cuong che", "thuong mai dien tu"]):
            topic_boost -= 0.55
        if topic_profile == "invoice_signature" and all(token not in normalized_title for token in ["hoa don", "dien tu", "quan ly thue", "123/2020", "119/2018", "68/2019"]):
            topic_boost -= 0.65
        if topic_profile == "social_insurance_penalty" and "bao hiem xa hoi" not in normalized_title and "xu phat" not in normalized_title:
            topic_boost -= 0.55
        if topic_profile == "copyright_registration" and all(token not in normalized_title for token in ["quyen tac gia", "so huu tri tue", "ban quyen"]):
            topic_boost -= 0.6

        if matched_rules and domain_match == 0.0 and lexical_overlap < 0.12 and title_match < 0.12:
            topic_boost -= 0.12
        if matched_rules and domain_value == "banking_law" and topic_boost < 0.0:
            topic_boost -= 0.05

        lexical_overlap = min(1.0, lexical_overlap)
        title_match = min(1.0, title_match)

        final_score = (
            float(base_score) * 0.18
            + lexical_overlap * 0.22
            + title_match * 0.18
            + citation_match * 0.06
            + domain_match * 0.14
            + topic_boost
        )
        final_score = max(0.0, float(final_score))
        return {
            "lexical_overlap": float(round(lexical_overlap, 4)),
            "title_match": float(round(title_match, 4)),
            "citation_match": float(round(citation_match, 4)),
            "domain_match": float(round(domain_match, 4)),
            "topic_boost": float(round(topic_boost, 4)),
            "final_score": float(round(final_score, 6)),
        }

    def _make_item(self, chunk_id: str, score: float, method: str) -> _SearchItem | None:
        chunk = self.chunk_by_id.get(chunk_id)
        if not chunk:
            return None
        metadata = {
            "doc_id": chunk.get("doc_id"),
            "domain": chunk.get("domain"),
            "doc_title": chunk.get("doc_title"),
            "article": chunk.get("article"),
            "clause": chunk.get("clause"),
            "citation": chunk.get("citation"),
            "source_url": chunk.get("source_url"),
        }
        return _SearchItem(
            chunk_id=chunk_id,
            score=float(score),
            retrieval_method=method,
            content=str(chunk.get("content") or ""),
            embedding_text=str(chunk.get("embedding_text") or chunk.get("content") or ""),
            metadata=metadata,
        )

    def _metadata_row_for_index(self, index: int) -> Dict[str, object] | None:
        metadata_row = self.metadata_by_index.get(int(index))
        if metadata_row:
            return metadata_row
        if 0 <= int(index) < len(self.metadata_rows):
            candidate = self.metadata_rows[int(index)]
            if isinstance(candidate, dict):
                return candidate
        return None

    def _dense_search(self, query: str, *, top_k: int, domain: str | Sequence[str] | None = None) -> List[_SearchItem]:
        candidate_limit = max(top_k * 20, top_k, 100)
        try:
            query_vector = np.array([self.embedding_model.embed_query(query)], dtype="float32")
            faiss.normalize_L2(query_vector)
            scores, indices = self.index.search(query_vector, candidate_limit)
        except Exception:
            return []

        items: List[_SearchItem] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            metadata_row = self._metadata_row_for_index(int(index))
            if not metadata_row or not self._allowed_domain(metadata_row, domain):
                continue
            item = self._make_item(str(metadata_row["chunk_id"]), float(score), "dense")
            if item:
                items.append(item)
            if len(items) >= candidate_limit:
                break
        return items

    def _sparse_search(self, query: str, *, top_k: int, domain: str | Sequence[str] | None = None) -> List[_SearchItem]:
        candidate_limit = max(top_k * 20, top_k, 100)
        try:
            query_tokens = tokenize_for_bm25(query)
            if not query_tokens:
                return []
            scores = self.bm25.get_scores(query_tokens)
        except Exception:
            return []

        ranked_items = nlargest(candidate_limit, scores.items(), key=lambda item: item[1])
        ranked_indices = [idx for idx, _score in ranked_items]
        ranked_scores = [score for _idx, score in ranked_items]
        normalized = self._normalize(ranked_scores)
        items: List[_SearchItem] = []
        for idx, score in zip(ranked_indices, normalized):
            if idx >= len(self.bm25_chunk_ids):
                continue
            chunk_id = self.bm25_chunk_ids[idx]
            chunk = self.chunk_by_id.get(chunk_id)
            if not chunk or not self._allowed_domain(chunk, domain):
                continue
            item = self._make_item(chunk_id, float(score), "sparse")
            if item:
                items.append(item)
            if len(items) >= candidate_limit:
                break
        return items

    def _lexical_fallback(self, query: str, *, top_k: int, domain: str | Sequence[str] | None = None) -> List[_SearchItem]:
        query_tokens = set(tokenize_for_bm25(query))
        scored: List[tuple[float, Dict[str, object]]] = []
        for chunk in self.chunks:
            if not self._allowed_domain(chunk, domain):
                continue
            text = self._combined_text(chunk)
            tokens = set(tokenize_for_bm25(text))
            if not tokens:
                continue
            score = len(query_tokens & tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
            if score <= 0.0 and not query_tokens:
                score = 0.0
            scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        items: List[_SearchItem] = []
        for score, chunk in scored[:top_k]:
            item = self._make_item(str(chunk.get("chunk_id")), float(score), "lexical_fallback")
            if item:
                items.append(item)
        return items

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        domain: str | Sequence[str] | None = None,
        preferred_domains: Sequence[str] | None = None,
    ) -> List[Dict[str, object]]:
        mode = os.getenv("R2AI_RETRIEVAL_MODE", "hybrid").strip().lower()
        dense = self._dense_search(query, top_k=top_k, domain=domain) if mode in {"dense", "hybrid"} else []
        sparse = self._sparse_search(query, top_k=top_k, domain=domain) if mode in {"sparse", "hybrid"} else []
        if preferred_domains is None:
            try:
                from src.retrieval.query_router import detect_domains

                preferred_domains = detect_domains(query)
            except Exception:
                preferred_domains = None

        fused: Dict[str, Dict[str, object]] = {}
        for item in dense:
            fused[item.chunk_id] = item.to_dict()
            fused[item.chunk_id]["dense_score"] = float(item.score)
            fused[item.chunk_id]["bm25_score"] = 0.0
            fused[item.chunk_id]["score"] = self.alpha * item.score
            fused[item.chunk_id]["retrieval_score"] = fused[item.chunk_id]["score"]
        for item in sparse:
            existing = fused.get(item.chunk_id)
            if existing is None:
                fused[item.chunk_id] = item.to_dict()
                fused[item.chunk_id]["dense_score"] = 0.0
                fused[item.chunk_id]["bm25_score"] = float(item.score)
                fused[item.chunk_id]["score"] = (1.0 - self.alpha) * item.score
                fused[item.chunk_id]["retrieval_score"] = fused[item.chunk_id]["score"]
                continue
            existing["retrieval_method"] = "hybrid"
            existing["bm25_score"] = float(item.score)
            existing["score"] = float(existing["score"]) + (1.0 - self.alpha) * item.score
            existing["retrieval_score"] = existing["score"]

        matched_rules = self._topic_rules(query)
        phrase_seed_score = 0.72
        seen_phrase_chunks = set(fused.keys())
        for rule in matched_rules:
            phrase_pool = list(rule.get("required_phrases", [])) + list(rule.get("title_phrases", []))
            for phrase in phrase_pool:
                normalized_phrase = _normalize_plain(str(phrase))
                if not normalized_phrase:
                    continue
                phrase_hits = self._phrase_hits(str(phrase), limit=8)
                for chunk_id in phrase_hits[:3]:
                    chunk = self.chunk_by_id.get(chunk_id)
                    if not chunk:
                        continue
                    chunk_domain = str(chunk.get("domain") or "")
                    if preferred_domains and chunk_domain and chunk_domain not in set(str(domain) for domain in preferred_domains):
                        if chunk_domain not in {"business_law", "investment_law"}:
                            continue
                    existing = fused.get(chunk_id)
                    if existing is not None:
                        boosted_score = max(float(existing.get("score") or 0.0), phrase_seed_score) + 0.08
                        existing["score"] = boosted_score
                        existing["retrieval_score"] = boosted_score
                        existing["retrieval_method"] = "phrase_boost"
                    else:
                        fused[chunk_id] = {
                            "chunk_id": chunk_id,
                            "score": phrase_seed_score,
                            "retrieval_score": phrase_seed_score,
                            "retrieval_method": "phrase_guard",
                            "content": str(chunk.get("content") or ""),
                            "embedding_text": str(chunk.get("embedding_text") or chunk.get("content") or ""),
                            "metadata": {
                                "doc_id": chunk.get("doc_id"),
                                "domain": chunk.get("domain"),
                                "doc_title": chunk.get("doc_title"),
                                "article": chunk.get("article"),
                                "clause": chunk.get("clause"),
                                "citation": chunk.get("citation"),
                                "source_url": chunk.get("source_url"),
                                "legal_path": chunk.get("legal_path"),
                            },
                        }
                    seen_phrase_chunks.add(chunk_id)

        enriched: List[Dict[str, object]] = []
        for row in fused.values():
            chunk_id = str(row.get("chunk_id") or "")
            chunk = self.chunk_by_id.get(chunk_id)
            if not chunk:
                continue
            adjustments = self._score_context(query, chunk, float(row.get("score") or 0.0), preferred_domains=preferred_domains)
            row.update(adjustments)
            row["metadata"] = {
                "doc_id": chunk.get("doc_id"),
                "domain": chunk.get("domain"),
                "doc_title": chunk.get("doc_title"),
                "article": chunk.get("article"),
                "clause": chunk.get("clause"),
                "citation": chunk.get("citation"),
                "source_url": chunk.get("source_url"),
                "legal_path": chunk.get("legal_path"),
            }
            row["score"] = float(row["final_score"])
            row["retrieval_score"] = float(row["final_score"])
            enriched.append(row)

        ranked = sorted(enriched, key=lambda row: float(row["final_score"]), reverse=True)
        if ranked:
            return ranked[:top_k]

        if domain is None:
            fallback = self._lexical_fallback(query, top_k=top_k, domain=None)
            fallback_rows = [item.to_dict() for item in fallback]
            for row in fallback_rows:
                chunk = self.chunk_by_id.get(str(row.get("chunk_id") or ""))
                if chunk:
                    row.update(self._score_context(query, chunk, float(row.get("score") or 0.0), preferred_domains=preferred_domains))
                    row["metadata"] = {
                        "doc_id": chunk.get("doc_id"),
                        "domain": chunk.get("domain"),
                        "doc_title": chunk.get("doc_title"),
                        "article": chunk.get("article"),
                        "clause": chunk.get("clause"),
                        "citation": chunk.get("citation"),
                        "source_url": chunk.get("source_url"),
                        "legal_path": chunk.get("legal_path"),
                    }
                    row["score"] = float(row["final_score"])
                    row["retrieval_score"] = float(row["final_score"])
            return sorted(fallback_rows, key=lambda row: float(row.get("final_score") or row.get("score") or 0.0), reverse=True)[:top_k]

        return []


def _cli() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Search ingestion index with hybrid dense + BM25 retrieval.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--domain", action="append", default=None)
    args = parser.parse_args()

    retriever = HybridRetriever()
    results = retriever.search(args.query, top_k=args.top_k, domain=args.domain or None)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
