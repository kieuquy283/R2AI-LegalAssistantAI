from __future__ import annotations

import argparse
import json
import re
import unicodedata
from typing import Dict, List


ARTICLE_RE = re.compile(r"dieu\s+\d+[a-z]?", re.IGNORECASE)


def _normalize(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    plain = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", plain).strip()


class GroundingValidator:
    def validate(
        self,
        *,
        query: str,
        answer: str,
        citations: List[Dict[str, object]],
        contexts: List[Dict[str, object]],
    ) -> Dict[str, object]:
        normalized_answer = _normalize(answer)
        citation_count = len(citations or [])
        warnings: List[str] = []
        unsupported_claims: List[str] = []

        has_answer = bool((answer or "").strip())
        has_citation = citation_count > 0
        missing_citation = bool(contexts) and not has_citation

        if not has_answer:
            warnings.append("empty_answer")
        if missing_citation:
            warnings.append("missing_citation_for_non_empty_context")

        context_refs = set()
        context_sources = set()
        for context in contexts or []:
            metadata = dict(context.get("metadata") or {})
            article = _normalize(str(metadata.get("article") or ""))
            citation = _normalize(str(metadata.get("citation") or ""))
            source_url = str(metadata.get("source_url") or "")
            if article:
                context_refs.add(article)
            if citation:
                context_refs.add(citation)
            if source_url:
                context_sources.add(source_url)

        linked_citations = 0
        for citation in citations or []:
            article = _normalize(str(citation.get("article") or ""))
            source_url = str(citation.get("source_url") or "")
            if (not article or article in context_refs) and (not source_url or source_url in context_sources):
                linked_citations += 1
        if has_citation and linked_citations < citation_count:
            warnings.append("citation_not_fully_linked_to_context")

        mentioned_articles = {match.group(0).strip() for match in ARTICLE_RE.finditer(normalized_answer)}
        for article in sorted(mentioned_articles):
            if article not in context_refs:
                unsupported_claims.append(article)
        if unsupported_claims:
            warnings.append("unsupported_legal_reference")

        if not contexts and "chua du can cu" not in normalized_answer:
            warnings.append("empty_context_without_insufficient_basis_notice")

        is_grounded = has_answer and not unsupported_claims and not missing_citation and linked_citations == citation_count
        return {
            "is_grounded": is_grounded,
            "has_citation": has_citation,
            "citation_count": citation_count,
            "missing_citation": missing_citation,
            "unsupported_claims": unsupported_claims,
            "warnings": warnings,
        }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Validate whether an answer is grounded in retrieved context.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--citations-json", required=True)
    parser.add_argument("--contexts-json", required=True)
    args = parser.parse_args()
    validator = GroundingValidator()
    result = validator.validate(
        query=args.query,
        answer=args.answer,
        citations=json.loads(args.citations_json),
        contexts=json.loads(args.contexts_json),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
