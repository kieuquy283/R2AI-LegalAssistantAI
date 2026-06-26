from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from rag.config.runtime import get_retrieval_runtime_config
from src.generation.answer_generator import AnswerGenerator
from src.generation.grounding_validator import GroundingValidator
from src.retrieval.retrieval_pipeline import RetrievalPipeline


_LEGAL_QA_PIPELINE_INSTANCE: LegalQAPipeline | None = None


class LegalQAPipeline:
    def __new__(cls) -> LegalQAPipeline:
        global _LEGAL_QA_PIPELINE_INSTANCE
        if _LEGAL_QA_PIPELINE_INSTANCE is not None:
            return _LEGAL_QA_PIPELINE_INSTANCE
        instance = super().__new__(cls)
        _LEGAL_QA_PIPELINE_INSTANCE = instance
        return instance

    def __init__(self) -> None:
        # Prevent re-initialization if singleton already initialized
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.runtime_config = get_retrieval_runtime_config()
        self.retrieval_pipeline = RetrievalPipeline()
        self.answer_generator = AnswerGenerator()
        self.grounding_validator = GroundingValidator()
        self.document_catalog, self.document_title_catalog = self._load_document_catalog()

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        global _LEGAL_QA_PIPELINE_INSTANCE
        _LEGAL_QA_PIPELINE_INSTANCE = None

    @staticmethod
    def _load_document_catalog() -> tuple[dict[str, dict], dict[str, dict]]:
        catalog: dict[str, dict] = {}
        title_catalog: dict[str, dict] = {}
        base_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
        overridden = os.getenv("R2AI_DOCUMENTS_PATH", "").strip()
        if overridden:
            candidates = [Path(overridden).name]
            base_dir = Path(overridden).parent
        else:
            candidates = []
        backend = str(os.getenv("RETRIEVAL_BACKEND", "faiss")).strip().lower()
        if not candidates:
            candidates = ["merged_documents.jsonl", "documents.jsonl"] if backend == "qdrant" else ["documents.jsonl", "merged_documents.jsonl"]
        for filename in candidates:
            documents_path = base_dir / filename
            if not documents_path.exists():
                continue
            with documents_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    doc_id = str(row.get("doc_id") or "").strip()
                    if doc_id:
                        catalog[doc_id] = row
                    doc_title = re.sub(r"\s+", " ", str(row.get("doc_title") or "").strip().lower())
                    if doc_title:
                        title_catalog[doc_title] = row
            if catalog or title_catalog:
                break
        return catalog, title_catalog

    @staticmethod
    def _normalize_space(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip(" ,;:-")

    def _lookup_document(self, metadata: dict) -> dict:
        doc_id = str(metadata.get("doc_id") or "").strip()
        if doc_id and doc_id in self.document_catalog:
            return dict(self.document_catalog[doc_id])
        doc_title = re.sub(r"\s+", " ", str(metadata.get("doc_title") or "").strip().lower())
        if doc_title and doc_title in self.document_title_catalog:
            return dict(self.document_title_catalog[doc_title])
        return {}

    @staticmethod
    def _context_metadata(context: dict) -> dict:
        merged = dict(context)
        nested = dict(context.get("metadata") or {})
        merged.update({key: value for key, value in nested.items() if value not in (None, "", [])})
        return merged

    def _format_doc_name(self, metadata: dict) -> tuple[str, str]:
        document = self._lookup_document(metadata)
        doc_type = self._normalize_space(str(document.get("doc_type") or metadata.get("doc_type") or ""))
        doc_number = self._normalize_space(str(document.get("doc_number") or metadata.get("doc_number") or ""))
        doc_title = self._normalize_space(str(document.get("doc_title") or metadata.get("doc_title") or metadata.get("doc_id") or ""))

        summary = doc_title
        if doc_type and summary.lower().startswith(doc_type.lower()):
            summary = summary[len(doc_type) :].strip(" ,:-")
        if doc_number:
            summary = re.sub(rf"(^|,\s*)số\s+{re.escape(doc_number)}", " ", summary, flags=re.IGNORECASE)
            summary = re.sub(rf"\b{re.escape(doc_number)}\b", " ", summary, flags=re.IGNORECASE)
        summary = re.sub(r"^của\s+[^,]+?\s+về\s+việc\s+", "", summary, flags=re.IGNORECASE)
        summary = re.sub(r"^của\s+[^,]+?\s+về\s+", "", summary, flags=re.IGNORECASE)
        summary = re.sub(r"^của\s+[^,]+?\s+", "", summary, flags=re.IGNORECASE)
        summary = self._normalize_space(summary)

        if doc_type and doc_number and summary:
            return doc_number, f"{doc_type} {doc_number} {summary}"
        if doc_type and doc_number:
            return doc_number, f"{doc_type} {doc_number}"
        if doc_number and doc_title:
            return doc_number, doc_title
        if doc_title:
            code = doc_number or self._normalize_space(str(metadata.get("doc_id") or ""))
            return code, doc_title
        return self._normalize_space(str(metadata.get("doc_id") or "")), self._normalize_space(str(metadata.get("doc_id") or ""))

    def _context_quality(self, question: str, contexts: list[dict], route: str = "") -> dict:
        if not contexts:
            return {"is_relevant": False, "reason": "no_contexts", "top_score": 0.0}
        has_score_signal = any(
            "final_score" in context or "score" in context or "rerank_score" in context
            for context in contexts
        )
        question_text = self._normalize_space(question).lower()
        top = self._context_metadata(contexts[0])
        top_score = float(top.get("final_score") or top.get("score") or 0.0)
        lexical_overlap = float(top.get("lexical_overlap") or top.get("rerank_score") or 0.0)
        title_match = float(top.get("title_match") or 0.0)
        domain_match = float(top.get("domain_match") or 0.0)
        topic_boost = float(top.get("topic_boost") or 0.0)
        has_title_signal = bool(
            str(top.get("doc_title") or "").strip()
            and (
                str(top.get("article") or "").strip()
                or str(top.get("citation") or "").strip()
            )
        )
        if not has_score_signal and any(str(self._context_metadata(context).get("doc_title") or "").strip() for context in contexts):
            return {"is_relevant": True, "reason": None, "top_score": 0.2}
        if top_score < 0.08:
            route = str(route or "").upper()
            tax_question_signal = any(
                keyword in question_text
                for keyword in ("thuế", "tài chính", "kế toán", "miễn thuế", "giảm thuế", "đăng ký thuế")
            )
            if route in {"PARENT_CONTEXT", "SIMPLE_VECTOR"} and tax_question_signal:
                for context in contexts[:5]:
                    metadata = self._context_metadata(context)
                    if str(metadata.get("domain") or "").strip().lower() != "tax":
                        continue
                    context_score = float(context.get("final_score") or context.get("score") or 0.0)
                    context_lexical = float(context.get("lexical_overlap") or context.get("rerank_score") or 0.0)
                    has_context_signal = bool(
                        str(metadata.get("doc_title") or "").strip()
                        and (
                            str(metadata.get("article") or "").strip()
                            or str(metadata.get("citation") or "").strip()
                        )
                    )
                    if has_context_signal and context_score >= 0.06 and context_lexical >= 0.28:
                        return {"is_relevant": True, "reason": None, "top_score": context_score}
            if (
                route in {"PARENT_CONTEXT", "SIMPLE_VECTOR"}
                and str(top.get("domain") or "").strip().lower() == "tax"
                and has_title_signal
                and tax_question_signal
                and lexical_overlap >= 0.28
            ):
                return {"is_relevant": True, "reason": None, "top_score": top_score}
            return {"is_relevant": False, "reason": "top_score_too_low", "top_score": top_score}
        if top_score < 0.12:
            strong_signal = max(lexical_overlap, title_match, domain_match, topic_boost)
            route = str(route or "").upper()
            if route in {"PARENT_CONTEXT", "SIMPLE_VECTOR"} and has_title_signal and strong_signal >= 0.08:
                return {"is_relevant": True, "reason": None, "top_score": top_score}
            if route in {"CROSS_DOMAIN_CONTEXT", "LEGAL_GRAPH_CONTEXT"} and has_title_signal and strong_signal >= 0.10:
                return {"is_relevant": True, "reason": None, "top_score": top_score}
        if top_score < 0.22 and max(lexical_overlap, title_match, domain_match, topic_boost) < 0.08:
            return {"is_relevant": False, "reason": "weak_topic_alignment", "top_score": top_score}
        if lexical_overlap < 0.03 and title_match < 0.03 and domain_match == 0.0 and topic_boost <= 0.0 and top_score < 0.25:
            return {"is_relevant": False, "reason": "no_relevance_signal", "top_score": top_score}
        return {"is_relevant": True, "reason": None, "top_score": top_score}

    def _build_relevant_docs(self, contexts: list[dict]) -> list[str]:
        relevant_docs: list[str] = []
        seen: set[str] = set()
        citation_contexts = self._citation_contexts(contexts)
        for context in citation_contexts:
            metadata = self._context_metadata(context)
            doc_code, doc_name = self._format_doc_name(metadata)
            if not doc_code or not doc_name:
                continue
            ref = f"{doc_code}|{doc_name}"
            if ref in seen:
                continue
            seen.add(ref)
            relevant_docs.append(ref)
            if len(relevant_docs) >= self.runtime_config.max_docs:
                break
        return relevant_docs

    def _build_relevant_doc_details(self, contexts: list[dict]) -> list[dict]:
        doc_details: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for context in self._citation_contexts(contexts):
            metadata = self._context_metadata(context)
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
                    "doc_id": str(metadata.get("doc_id") or "").strip(),
                    "doc_title": doc_title,
                    "source_url": source_url,
                    "citation": citation,
                }
            )
            if len(doc_details) >= self.runtime_config.max_docs:
                break
        return doc_details

    def _build_citation_payload(self, contexts: list[dict]) -> list[dict]:
        citations: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for context in self._citation_contexts(contexts):
            metadata = self._context_metadata(context)
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
            if len(citations) >= self.runtime_config.max_contexts:
                break
        return citations

    def _build_relevant_articles(self, contexts: list[dict]) -> list[str]:
        relevant_articles: list[str] = []
        seen: set[str] = set()
        for context in self._citation_contexts(contexts):
            metadata = self._context_metadata(context)
            doc_code, doc_name = self._format_doc_name(metadata)
            article = str(metadata.get("article") or "").strip()
            if not doc_code or not doc_name or not article:
                continue
            ref = f"{doc_code}|{doc_name}|{article}"
            if ref in seen:
                continue
            seen.add(ref)
            relevant_articles.append(ref)
            if len(relevant_articles) >= self.runtime_config.max_articles:
                break
        return relevant_articles

    def _build_relevant_article_details(self, contexts: list[dict]) -> list[dict]:
        article_details: list[dict] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for context in self._citation_contexts(contexts):
            metadata = self._context_metadata(context)
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
                    "doc_id": str(metadata.get("doc_id") or "").strip(),
                    "doc_title": doc_title,
                    "article": article,
                    "clause": clause or None,
                    "citation": citation,
                    "source_url": source_url,
                }
            )
            if len(article_details) >= self.runtime_config.max_articles:
                break
        return article_details

    def _citation_contexts(self, contexts: list[dict]) -> list[dict]:
        if not contexts:
            return []
        strict: list[dict] = []
        for context in contexts:
            score = float(context.get("final_score") or context.get("score") or 0.0)
            if score >= self.runtime_config.citation_score_threshold:
                strict.append(context)
        if strict:
            return strict

        best = float(contexts[0].get("final_score") or contexts[0].get("score") or 0.0)
        relaxed_threshold = min(
            self.runtime_config.citation_score_threshold,
            max(0.08, best * 0.7),
        )
        relaxed: list[dict] = []
        for context in contexts:
            score = float(context.get("final_score") or context.get("score") or 0.0)
            metadata = self._context_metadata(context)
            has_signal = bool(
                str(metadata.get("doc_title") or "").strip()
                and (
                    str(metadata.get("article") or "").strip()
                    or str(metadata.get("citation") or "").strip()
                )
            )
            if score >= relaxed_threshold and has_signal:
                relaxed.append(context)
        return relaxed

    # Forbidden phrases when context exists - model should NOT say these
    _FORBIDDEN_PHRASES = [
        "context không cung cấp",
        "context khong cung cap",
        "chưa đủ căn cứ",
        "chưa đủ thông tin",
        "thiếu thông tin",
        "không đủ thông tin",
        "không đủ căn cứ",
        "chưa có đủ căn cứ",
        "không có đủ thông tin",
        "dữ liệu truy xuất không đủ",
        "chưa đủ dữ liệu",
    ]

    _FORBIDDEN_PATTERNS = [
        re.compile(r"(?i)context\s+(không|khong)\s+(cung cấp|cung cap|có|co)"),
        re.compile(r"(?i)chưa\s+(đủ|du)\s+(căn cứ|can cu|thông tin|thong tin|dữ liệu|du lieu)"),
        re.compile(r"(?i)thiếu\s+(thông tin|thong tin|căn cứ|can cu|dữ liệu|du lieu)"),
        re.compile(r"(?i)không\s+(đủ|du|co| có)\s+(thông tin|thong tin|căn cứ|can cu|dữ liệu|du lieu)"),
    ]

    def _validate_answer(self, answer_text: str, contexts: list[dict]) -> str:
        """Check if model incorrectly claims no context when context exists, and verify 4 mandatory sections."""
        if not contexts:
            return answer_text

        # 1. Check for 4 mandatory sections
        required_sections = [
            r"(?i)1\.\s*kết luận ngắn",
            r"(?i)2\.\s*căn cứ pháp luật",
            r"(?i)3\.\s*phân tích áp dụng",
            r"(?i)4\.\s*việc sme nên làm"
        ]
        missing_sections = []
        for section in required_sections:
            if not re.search(section, answer_text):
                missing_sections.append(section)

        if missing_sections:
            print(f"[QAPipeline] Answer is missing mandatory sections: {missing_sections}. Triggering fallback...")
            citations = self._build_citation_payload(contexts)
            return self.answer_generator._fallback_answer(contexts=contexts, citations=citations)

        # 2. Check for forbidden phrases (model claiming no context when context exists)
        # Only check short answers (<300 chars) or answers that explicitly deny having context
        if len(answer_text) > 300:
            # For longer answers, only check explicit forbidden phrases, not patterns
            answer_lower = answer_text.lower()
            matched_phrase = None
            for phrase in self._FORBIDDEN_PHRASES:
                if phrase in answer_lower:
                    matched_phrase = phrase
                    break

            if matched_phrase:
                print(f"[QAPipeline] Answer contained forbidden phrase '{matched_phrase}' but context exists. Regenerating fallback...")
                citations = self._build_citation_payload(contexts)
                return self.answer_generator._fallback_answer(contexts=contexts, citations=citations)
            return answer_text

        # For short answers, check both phrases and patterns
        has_forbidden = False
        matched_phrase = None
        answer_lower = answer_text.lower()
        for phrase in self._FORBIDDEN_PHRASES:
            if phrase in answer_lower:
                has_forbidden = True
                matched_phrase = phrase
                break

        if not has_forbidden:
            for pattern in self._FORBIDDEN_PATTERNS:
                if pattern.search(answer_text):
                    has_forbidden = True
                    matched_phrase = "pattern_match"
                    break

        if has_forbidden:
            print(f"[QAPipeline] Answer contained forbidden phrase '{matched_phrase}' but context exists. Regenerating fallback...")
            citations = self._build_citation_payload(contexts)
            return self.answer_generator._fallback_answer(contexts=contexts, citations=citations)

        return answer_text

    def _extract_citations_from_answer(self, answer_text: str) -> list[dict]:
        """Extract Điều/Khoản/Điểm citations from answer text."""
        citations = []
        # Pattern: Điều X, Khoản Y, Điểm Z
        pattern = re.compile(
            r"(Điều\s+\d+[a-zA-Z]?\s*(?:,\s*Khoản\s+\d+[a-zA-Z]?)?\s*(?:,\s*Điểm\s+\d+[a-zA-Z]?)?)"
        )
        matches = pattern.findall(answer_text)
        for match in set(matches):
            citations.append({"citation": match.strip()})
        return citations

    def answer(
        self,
        question: str,
        *,
        include_grounding: bool = True,
        use_llm: bool = True,
    ) -> dict:
        retrieval_result = self.retrieval_pipeline.run(question)
        raw_final_contexts = list(retrieval_result.get("final_contexts") or [])
        quality = self._context_quality(question, raw_final_contexts, route=str(retrieval_result.get("route") or ""))
        final_contexts = raw_final_contexts if quality["is_relevant"] else []
        answer_retrieval_result = dict(retrieval_result)
        answer_retrieval_result["final_contexts"] = final_contexts

        # Optional: disable answer generation entirely
        disable_answer = os.getenv("R2AI_DISABLE_ANSWER", "").strip().lower() in {"1", "true", "yes"}
        if disable_answer:
            return {
                "question": question,
                "route": retrieval_result["route"],
                "domains": retrieval_result["domains"],
                "answer": "",
                "citations": [],
                "relevant_docs": [],
                "relevant_doc_details": [],
                "relevant_articles": [],
                "relevant_article_details": [],
                "grounding": None,
                "low_confidence": not quality["is_relevant"],
                "low_confidence_reason": quality["reason"],
                "retrieved_chunks": retrieval_result["seed_chunks"],
                "seed_contexts": retrieval_result["seed_contexts"],
                "expanded_contexts": retrieval_result["expanded_contexts"],
                "final_contexts": final_contexts,
                "raw_final_contexts": raw_final_contexts,
                "answer_citations": [],
            }

        generated = self.answer_generator.generate(query=question, retrieval_result=answer_retrieval_result, use_llm=use_llm)
        citations = self._build_citation_payload(final_contexts)
        relevant_docs = self._build_relevant_docs(final_contexts)
        relevant_doc_details = self._build_relevant_doc_details(final_contexts)
        relevant_articles = self._build_relevant_articles(final_contexts)
        relevant_article_details = self._build_relevant_article_details(final_contexts)

        answer_text = str(generated.get("answer") or "")
        
        # Validate: if context exists but model says no context, regenerate
        if final_contexts:
            answer_text = self._validate_answer(answer_text, final_contexts)
        
        # Also check old fallback triggers
        if final_contexts and (
            "Chưa đủ căn cứ pháp lý" in answer_text
            or "ChÆ°a Ä‘á»§ cÄƒn cá»© phÃ¡p lÃ½" in answer_text
        ):
            answer_text = self.answer_generator._fallback_answer(contexts=final_contexts, citations=citations)

        # Extract citations from answer text for verification
        answer_citations = self._extract_citations_from_answer(answer_text)

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
            "answer_citations": answer_citations,
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
