from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List

from src.retrieval.query_router import (
    CROSS_DOMAIN_CONTEXT,
    LEGAL_GRAPH_CONTEXT,
    MULTI_DOMAIN_COMPLEX,
    PARENT_CONTEXT,
    SIMPLE_VECTOR,
    detect_domains,
)


MIN_TOP_SCORE = 0.55
MIN_SCORE_GAP = 0.03
MIN_SEED_CHUNKS = 2


def _normalize(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


class ConfidenceChecker:
    def __init__(
        self,
        *,
        min_top_score: float = MIN_TOP_SCORE,
        min_score_gap: float = MIN_SCORE_GAP,
        min_seed_chunks: int = MIN_SEED_CHUNKS,
    ) -> None:
        self.min_top_score = float(min_top_score)
        self.min_score_gap = float(min_score_gap)
        self.min_seed_chunks = int(min_seed_chunks)
        self.taxonomy = json.loads(Path("data/sources/domain_taxonomy.json").read_text(encoding="utf-8")) if Path("data/sources/domain_taxonomy.json").exists() else {}

    def _has_legal_signal(self, texts: Iterable[str]) -> bool:
        joined = _normalize(" ".join(texts))
        legal_markers = [
            "dieu ",
            "khoan ",
            "diem ",
            "luat ",
            "nghi dinh",
            "can cu",
            "trach nhiem",
            "xu phat",
            "muc phat",
            "hieu luc",
        ]
        return any(marker in joined for marker in legal_markers)

    def _contains_any(self, text: str, keywords: Iterable[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _recommended_route(
        self,
        *,
        normalized_query: str,
        current_route: str,
        domains: List[str],
        reasons: List[str],
    ) -> str:
        if self._contains_any(normalized_query, ["toan bo", "nhung viec gi", "can lam gi", "so sanh"]):
            return MULTI_DOMAIN_COMPLEX
        if len(domains) > 1 or "satellite_domain_mismatch" in reasons:
            if current_route == CROSS_DOMAIN_CONTEXT and self._contains_any(
                normalized_query,
                ["can lam gi", "nhung viec gi", "ho so", "trinh tu", "thu tuc"],
            ):
                return MULTI_DOMAIN_COMPLEX
            return CROSS_DOMAIN_CONTEXT
        if self._contains_any(
            normalized_query,
            [
                "sua doi",
                "bo sung",
                "thay the",
                "het hieu luc",
                "con hieu luc",
                "theo quy dinh tai",
                "can cu",
                "ngoai le",
                "tru truong hop",
            ],
        ):
            return LEGAL_GRAPH_CONTEXT
        if self._contains_any(normalized_query, ["bi phat", "xu phat", "muc phat", "hieu luc", "thu tuc", "ho so"]):
            return LEGAL_GRAPH_CONTEXT if current_route == PARENT_CONTEXT else CROSS_DOMAIN_CONTEXT
        if self._contains_any(
            normalized_query,
            ["dieu", "khoan", "diem", "doi tuong nao", "truong hop nao", "dieu kien", "ai khong duoc"],
        ):
            return PARENT_CONTEXT
        if current_route == SIMPLE_VECTOR:
            return PARENT_CONTEXT
        if current_route == PARENT_CONTEXT and "weak_score_gap" in reasons:
            return LEGAL_GRAPH_CONTEXT
        return current_route

    def check(
        self,
        *,
        query: str,
        route_result: Dict[str, object],
        seed_chunks: Iterable[Dict[str, object]],
    ) -> Dict[str, object]:
        seed_list = list(seed_chunks)
        normalized_query = _normalize(query)
        scores = [float(chunk.get("score") or 0.0) for chunk in seed_list]
        top_score = scores[0] if scores else 0.0
        score_gap = top_score - scores[1] if len(scores) > 1 else 0.0
        top_metadata = [dict(chunk.get("metadata") or {}) for chunk in seed_list[:3]]
        top_texts = [
            " ".join(
                str(part or "")
                for part in [
                    chunk.get("content"),
                    chunk.get("metadata", {}).get("doc_title"),
                    chunk.get("metadata", {}).get("citation"),
                    chunk.get("metadata", {}).get("article"),
                ]
                if str(part or "").strip()
            )
            for chunk in seed_list[:3]
        ]
        domains = list(dict.fromkeys([str(domain) for domain in detect_domains(query)]))
        satellite_domains = {domain for domain in domains if domain != "business_law"}

        reasons: List[str] = []
        if top_score < self.min_top_score:
            reasons.append("top_score_below_threshold")
        if len(scores) < self.min_seed_chunks:
            reasons.append("insufficient_seed_chunks")
        elif score_gap < self.min_score_gap:
            reasons.append("weak_score_gap")
        if not any(metadata.get("citation") and metadata.get("article") for metadata in top_metadata):
            reasons.append("missing_legal_citation")
        if not any(metadata.get("source_url") for metadata in top_metadata):
            reasons.append("missing_source_url")
        if top_texts and not self._has_legal_signal(top_texts):
            reasons.append("missing_legal_signal")
        if satellite_domains:
            combined_text = _normalize(" ".join(top_texts))
            matching_domains = set()
            for domain in satellite_domains:
                keywords = [str(keyword) for keyword in self.taxonomy.get(domain, {}).get("keywords", [])]
                if any(_normalize(keyword) in combined_text for keyword in keywords):
                    matching_domains.add(domain)
            if not matching_domains:
                reasons.append("satellite_domain_mismatch")
        if self._contains_any(normalized_query, ["bi phat", "xu phat", "muc phat"]) and not self._contains_any(
            _normalize(" ".join(top_texts)),
            ["phat", "xu phat", "vi pham", "che tai", "nghi dinh"],
        ):
            reasons.append("penalty_context_mismatch")
        if self._contains_any(normalized_query, ["thu tuc", "ho so", "trinh tu"]) and not self._contains_any(
            _normalize(" ".join(top_texts)),
            ["thu tuc", "ho so", "trinh tu", "dang ky"],
        ):
            reasons.append("procedure_context_mismatch")
        if self._contains_any(normalized_query, ["hieu luc", "het hieu luc", "con hieu luc"]) and not self._contains_any(
            _normalize(" ".join(top_texts)),
            ["hieu luc", "het hieu luc", "ap dung"],
        ):
            reasons.append("effective_status_mismatch")

        penalties = {
            "top_score_below_threshold": 0.22,
            "insufficient_seed_chunks": 0.16,
            "weak_score_gap": 0.12,
            "missing_legal_citation": 0.14,
            "missing_source_url": 0.08,
            "missing_legal_signal": 0.12,
            "satellite_domain_mismatch": 0.15,
            "penalty_context_mismatch": 0.12,
            "procedure_context_mismatch": 0.12,
            "effective_status_mismatch": 0.12,
        }
        confidence_score = max(0.0, min(1.0, 1.0 - sum(penalties.get(reason, 0.08) for reason in reasons)))
        recommended_route = self._recommended_route(
            normalized_query=normalized_query,
            current_route=str(route_result.get("route") or SIMPLE_VECTOR),
            domains=domains,
            reasons=reasons,
        )
        is_confident = confidence_score >= 0.55 and not reasons
        should_escalate = not is_confident and recommended_route != str(route_result.get("route") or SIMPLE_VECTOR)
        return {
            "is_confident": is_confident,
            "confidence_score": round(confidence_score, 4),
            "reasons": reasons,
            "recommended_route": recommended_route,
            "should_escalate": should_escalate,
            "top_score": round(top_score, 4),
            "score_gap": round(score_gap, 4),
            "seed_chunk_count": len(seed_list),
        }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Check retrieval confidence and recommend route escalation.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--route", default=SIMPLE_VECTOR)
    parser.add_argument("--seed-json", required=True, help="JSON array of seed chunk objects.")
    args = parser.parse_args()
    checker = ConfidenceChecker()
    result = checker.check(
        query=args.query,
        route_result={"route": args.route, "domains": detect_domains(args.query)},
        seed_chunks=json.loads(args.seed_json),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
