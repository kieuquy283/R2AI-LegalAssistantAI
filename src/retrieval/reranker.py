from __future__ import annotations

import argparse
import json
import unicodedata
from typing import Dict, List, Sequence

from rag.modules.retrieval.utils import tokenize_for_bm25
from src.retrieval.hybrid_retriever import TOPIC_RULES, _normalize_plain


RELATION_BOOSTS = {
    "seed": 0.20,
    "parent": 0.15,
    "explicit_reference": 0.18,
    "same_article": 0.12,
    "cross_domain": 0.10,
    "neighbor": 0.05,
}


class Reranker:
    def _combined_text(self, context: Dict[str, object]) -> str:
        metadata = dict(context.get("metadata") or {})
        parts = [
            metadata.get("doc_title"),
            metadata.get("domain"),
            metadata.get("legal_path"),
            metadata.get("citation"),
            metadata.get("article"),
            metadata.get("clause"),
            context.get("content"),
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

    def keyword_overlap_score(self, query: str, text: str) -> float:
        query_tokens = set(tokenize_for_bm25(query))
        text_tokens = set(tokenize_for_bm25(text))
        if not query_tokens or not text_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)

    def _score_context(self, query: str, context: Dict[str, object]) -> Dict[str, float]:
        metadata = dict(context.get("metadata") or {})
        combined_text = self._combined_text(context)
        title_text = " ".join(
            str(part or "")
            for part in [metadata.get("doc_title"), metadata.get("citation"), metadata.get("legal_path")]
            if str(part or "").strip()
        )

        retrieval_score = float(context.get("retrieval_score") or context.get("score") or 0.0)
        lexical_overlap = self.keyword_overlap_score(query, combined_text)
        title_match = self.keyword_overlap_score(query, title_text)
        citation_match = self.keyword_overlap_score(query, str(metadata.get("citation") or ""))
        preferred_domains = set()
        normalized_query = _normalize_plain(query)
        topic_boost = 0.0

        for rule in self._topic_rules(query):
            preferred_domains.update(str(domain) for domain in rule.get("preferred_domains", []))
            normalized_text = _normalize_plain(combined_text)
            normalized_title = _normalize_plain(title_text)
            required_hits = [phrase for phrase in rule["required_phrases"] if phrase in normalized_text]
            title_phrases = [phrase for phrase in rule["title_phrases"] if phrase in normalized_title or phrase in normalized_text]
            keyword_hits = [keyword for keyword in rule["keywords"] if keyword in normalized_text]
            if required_hits or title_phrases:
                topic_boost += float(rule["boost"]) * min(3.0, 1.0 * len(required_hits) + 0.6 * len(title_phrases))
                title_match += min(0.45, 0.12 * len(title_phrases))
                lexical_overlap += min(0.15, 0.03 * len(required_hits))
            elif keyword_hits:
                topic_boost += float(rule["boost"]) * 0.30
            if keyword_hits:
                topic_boost += min(0.12, 0.03 * len(keyword_hits))
            if required_hits:
                topic_boost += min(0.18, 0.04 * len(required_hits))
            if preferred_domains and str(metadata.get("domain") or "") in preferred_domains:
                topic_boost += 0.04

        domain_value = str(metadata.get("domain") or "")
        domain_match = 0.0
        if preferred_domains and domain_value in preferred_domains and domain_value != "business_law":
            domain_match = 1.0
        elif domain_value == "business_law" and preferred_domains == {"business_law"}:
            domain_match = 0.25
        relation_boost = RELATION_BOOSTS.get(str(context.get("context_type")), 0.0)
        if metadata.get("citation"):
            relation_boost += 0.02
        if metadata.get("source_url"):
            relation_boost += 0.02

        if lexical_overlap < 0.08 and title_match < 0.08 and domain_match == 0.0 and topic_boost <= 0.0:
            topic_boost -= 0.12

        final_score = (
            retrieval_score * 0.18
            + lexical_overlap * 0.22
            + title_match * 0.24
            + citation_match * 0.06
            + domain_match * 0.12
            + relation_boost * 0.08
            + topic_boost
        )
        return {
            "retrieval_score": retrieval_score,
            "lexical_overlap": round(lexical_overlap, 4),
            "title_match": round(title_match, 4),
            "citation_match": round(citation_match, 4),
            "domain_match": round(domain_match, 4),
            "topic_boost": round(topic_boost, 4),
            "relation_boost": round(relation_boost, 4),
            "final_score": round(max(0.0, final_score), 6),
        }

    def rerank(self, query: str, contexts: Sequence[Dict[str, object]], *, max_contexts: int = 5) -> List[Dict[str, object]]:
        deduped: Dict[str, Dict[str, object]] = {}
        for context in contexts:
            deduped[str(context["chunk_id"])] = dict(context)

        ranked: List[Dict[str, object]] = []
        for context in deduped.values():
            scores = self._score_context(query, context)
            context["retrieval_score"] = scores["retrieval_score"]
            context["rerank_score"] = scores["lexical_overlap"]
            context["lexical_overlap"] = scores["lexical_overlap"]
            context["title_match"] = scores["title_match"]
            context["citation_match"] = scores["citation_match"]
            context["domain_match"] = scores["domain_match"]
            context["topic_boost"] = scores["topic_boost"]
            context["relation_boost"] = scores["relation_boost"]
            context["final_score"] = scores["final_score"]
            if scores["final_score"] < 0.12 and scores["lexical_overlap"] < 0.05 and scores["title_match"] < 0.05:
                continue
            ranked.append(context)

        ranked.sort(key=lambda item: float(item["final_score"]), reverse=True)
        return ranked[:max_contexts]


def _cli() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Heuristic reranker for expanded legal contexts.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    demo = [
        {
            "chunk_id": "a",
            "content": "Điều 17. Người không được thành lập doanh nghiệp...",
            "retrieval_score": 0.7,
            "context_type": "seed",
            "metadata": {"source_url": "x", "citation": "Luật Doanh nghiệp, Điều 17"},
        },
        {
            "chunk_id": "b",
            "content": "Tin liên quan...",
            "retrieval_score": 0.6,
            "context_type": "neighbor",
            "metadata": {},
        },
    ]
    print(json.dumps(Reranker().rerank(args.query, demo), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
