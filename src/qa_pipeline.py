from __future__ import annotations

import argparse
import json
import sys

from src.generation.answer_generator import AnswerGenerator
from src.generation.grounding_validator import GroundingValidator
from src.retrieval.retrieval_pipeline import RetrievalPipeline


class LegalQAPipeline:
    def __init__(self) -> None:
        self.retrieval_pipeline = RetrievalPipeline()
        self.answer_generator = AnswerGenerator()
        self.grounding_validator = GroundingValidator()

    def _context_quality(self, contexts: list[dict]) -> dict:
        if not contexts:
            return {"is_relevant": False, "reason": "no_contexts", "top_score": 0.0}
        has_score_signal = any(
            "final_score" in context or "score" in context or "rerank_score" in context
            for context in contexts
        )
        top = dict(contexts[0])
        top_score = float(top.get("final_score") or top.get("score") or 0.0)
        lexical_overlap = float(top.get("lexical_overlap") or top.get("rerank_score") or 0.0)
        title_match = float(top.get("title_match") or 0.0)
        domain_match = float(top.get("domain_match") or 0.0)
        topic_boost = float(top.get("topic_boost") or 0.0)
        if not has_score_signal and any(str(dict(context.get("metadata") or {}).get("doc_title") or "").strip() for context in contexts):
            return {"is_relevant": True, "reason": None, "top_score": 0.2}
        if top_score < 0.12:
            return {"is_relevant": False, "reason": "top_score_too_low", "top_score": top_score}
        if top_score < 0.22 and max(lexical_overlap, title_match, domain_match, topic_boost) < 0.08:
            return {"is_relevant": False, "reason": "weak_topic_alignment", "top_score": top_score}
        if lexical_overlap < 0.03 and title_match < 0.03 and domain_match == 0.0 and topic_boost <= 0.0 and top_score < 0.25:
            return {"is_relevant": False, "reason": "no_relevance_signal", "top_score": top_score}
        return {"is_relevant": True, "reason": None, "top_score": top_score}

    def _build_relevant_docs(self, contexts: list[dict]) -> list[str]:
        relevant_docs: list[str] = []
        seen: set[str] = set()
        for context in contexts:
            metadata = dict(context.get("metadata") or {})
            doc_id = str(metadata.get("doc_id") or "").strip()
            doc_title = str(metadata.get("doc_title") or doc_id).strip()
            if not doc_title:
                continue
            ref = f"{doc_id or doc_title}|{doc_title}"
            if ref in seen:
                continue
            seen.add(ref)
            relevant_docs.append(ref)
        return relevant_docs

    def _build_relevant_doc_details(self, contexts: list[dict]) -> list[dict]:
        doc_details: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for context in contexts:
            metadata = dict(context.get("metadata") or {})
            doc_title = str(metadata.get("doc_title") or metadata.get("doc_id") or "").strip()
            source_url = str(metadata.get("source_url") or "").strip()
            citation = str(metadata.get("citation") or doc_title).strip()
            if not doc_title and not source_url and not citation:
                continue
            key = (doc_title, source_url)
            if key in seen:
                continue
            seen.add(key)
            doc_details.append(
                {
                    "doc_title": doc_title,
                    "source_url": source_url,
                    "citation": citation,
                }
            )
        return doc_details

    def _build_citation_payload(self, contexts: list[dict]) -> list[dict]:
        citations: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for context in contexts:
            metadata = dict(context.get("metadata") or {})
            key = (
                str(metadata.get("doc_title") or ""),
                str(metadata.get("article") or ""),
                str(metadata.get("source_url") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "doc_title": metadata.get("doc_title"),
                    "doc_id": metadata.get("doc_id"),
                    "article": metadata.get("article"),
                    "clause": metadata.get("clause"),
                    "citation": metadata.get("citation") or metadata.get("doc_title"),
                    "source_url": metadata.get("source_url"),
                }
            )
        return citations

    def _build_relevant_articles(self, contexts: list[dict]) -> list[str]:
        relevant_articles: list[str] = []
        seen: set[str] = set()
        for context in contexts:
            metadata = dict(context.get("metadata") or {})
            doc_id = str(metadata.get("doc_id") or "").strip()
            doc_title = str(metadata.get("doc_title") or doc_id).strip()
            article = str(metadata.get("article") or "").strip()
            if not doc_title or not article:
                continue
            ref = f"{doc_id or doc_title}|{doc_title}|{article}"
            if ref in seen:
                continue
            seen.add(ref)
            relevant_articles.append(ref)
        return relevant_articles

    def _build_relevant_article_details(self, contexts: list[dict]) -> list[dict]:
        article_details: list[dict] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for context in contexts:
            metadata = dict(context.get("metadata") or {})
            doc_title = str(metadata.get("doc_title") or metadata.get("doc_id") or "").strip()
            article = str(metadata.get("article") or "").strip()
            clause = str(metadata.get("clause") or "").strip()
            citation = str(metadata.get("citation") or doc_title).strip()
            source_url = str(metadata.get("source_url") or "").strip()
            if not doc_title and not article and not clause and not citation and not source_url:
                continue
            key = (doc_title, article, clause, citation, source_url)
            if key in seen:
                continue
            seen.add(key)
            article_details.append(
                {
                    "doc_title": doc_title,
                    "article": article,
                    "clause": clause or None,
                    "citation": citation,
                    "source_url": source_url,
                }
            )
        return article_details

    def answer(
        self,
        question: str,
        *,
        include_grounding: bool = True,
        use_llm: bool = True,
    ) -> dict:
        retrieval_result = self.retrieval_pipeline.run(question)
        raw_final_contexts = list(retrieval_result.get("final_contexts") or [])
        quality = self._context_quality(raw_final_contexts)
        final_contexts = raw_final_contexts if quality["is_relevant"] else []
        answer_retrieval_result = dict(retrieval_result)
        answer_retrieval_result["final_contexts"] = final_contexts
        generated = self.answer_generator.generate(query=question, retrieval_result=answer_retrieval_result, use_llm=use_llm)
        citations = self._build_citation_payload(final_contexts)
        relevant_docs = self._build_relevant_docs(final_contexts)
        relevant_doc_details = self._build_relevant_doc_details(final_contexts)
        relevant_articles = self._build_relevant_articles(final_contexts)
        relevant_article_details = self._build_relevant_article_details(final_contexts)

        answer_text = str(generated.get("answer") or "")
        if final_contexts and (
            "Chưa đủ căn cứ pháp lý" in answer_text
            or "ChÆ°a Ä‘á»§ cÄƒn cá»© phÃ¡p lÃ½" in answer_text
        ):
            answer_text = self.answer_generator._fallback_answer(contexts=final_contexts, citations=citations)

        grounding = None
        if include_grounding:
            grounding = self.grounding_validator.validate(
                query=question,
                answer=answer_text,
                citations=citations,
                contexts=final_contexts,
            )
        return {
            "question": question,
            "route": retrieval_result["route"],
            "domains": retrieval_result["domains"],
            "answer": answer_text,
            "citations": citations,
            "relevant_docs": relevant_docs,
            "relevant_doc_details": relevant_doc_details,
            "relevant_articles": relevant_articles,
            "relevant_article_details": relevant_article_details,
            "grounding": grounding,
            "low_confidence": not quality["is_relevant"],
            "low_confidence_reason": quality["reason"],
            "retrieved_chunks": retrieval_result["seed_chunks"],
            "seed_contexts": retrieval_result["seed_contexts"],
            "expanded_contexts": retrieval_result["expanded_contexts"],
            "final_contexts": final_contexts,
            "raw_final_contexts": raw_final_contexts,
        }


def _cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run full legal QA pipeline for one question.")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    print(json.dumps(LegalQAPipeline().answer(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
