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

    def answer(
        self,
        question: str,
        *,
        include_grounding: bool = True,
        use_llm: bool = True,
    ) -> dict:
        retrieval_result = self.retrieval_pipeline.run(question)
        generated = self.answer_generator.generate(query=question, retrieval_result=retrieval_result, use_llm=use_llm)
        final_contexts = list(retrieval_result.get("final_contexts") or [])
        citations = self._build_citation_payload(final_contexts)
        relevant_docs = self._build_relevant_docs(final_contexts)
        relevant_articles = self._build_relevant_articles(final_contexts)

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
            "relevant_articles": relevant_articles,
            "grounding": grounding,
            "retrieved_chunks": retrieval_result["seed_chunks"],
            "seed_contexts": retrieval_result["seed_contexts"],
            "expanded_contexts": retrieval_result["expanded_contexts"],
            "final_contexts": final_contexts,
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
