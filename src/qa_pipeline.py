from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

from rag.config.runtime import get_retrieval_runtime_config
from src.generation.answer_generator import AnswerGenerator
from src.generation.grounding_validator import GroundingValidator
from src.retrieval.retrieval_pipeline import RetrievalPipeline


_LEGAL_QA_PIPELINE_INSTANCE: LegalQAPipeline | None = None

# Semantic cache config
_CACHE_DIR = Path(os.getenv("R2AI_CACHE_DIR", "data/cache"))
_CACHE_PATH = _CACHE_DIR / "semantic_cache.pkl"
_CACHE_TTL = int(os.getenv("R2AI_CACHE_TTL", "3600"))  # 1 hour default
_CACHE_SIM_THRESHOLD = float(os.getenv("R2AI_CACHE_SIM_THRESHOLD", "0.95"))
_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-10)


class LegalQAPipeline:
    def __new__(cls) -> LegalQAPipeline:
        global _LEGAL_QA_PIPELINE_INSTANCE
        if _LEGAL_QA_PIPELINE_INSTANCE is not None:
            return _LEGAL_QA_PIPELINE_INSTANCE
        instance = super().__new__(cls)
        _LEGAL_QA_PIPELINE_INSTANCE = instance
        return instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.runtime_config = get_retrieval_runtime_config()
        self.retrieval_pipeline = RetrievalPipeline()
        self.answer_generator = AnswerGenerator()
        self.grounding_validator = GroundingValidator()
        self.document_catalog, self.document_title_catalog = self._load_document_catalog()
        self._semantic_cache: dict | None = None
        self._embeddings = None

    def _get_cache(self) -> dict:
        if self._semantic_cache is None:
            self._semantic_cache = _load_cache()
            # Prune expired entries
            now = time.time()
            expired = [k for k, v in self._semantic_cache.items() if now - v.get("_ts", 0) > _CACHE_TTL]
            for k in expired:
                del self._semantic_cache[k]
            if expired:
                print(f"[Cache] Pruned {len(expired)} expired entries")
        return self._semantic_cache

    def _flush_cache(self) -> None:
        if self._semantic_cache is not None:
            _save_cache(self._semantic_cache)

    def _embed_query(self, query: str) -> list[float]:
        """Embed query using the same model as retrieval pipeline."""
        if self._embeddings is None:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")
            print(f"[Cache] Loading embedding model: {model_name}")
            self._embeddings = SentenceTransformer(model_name, device="cpu")
        return self._embeddings.encode(query, normalize_embeddings=True).tolist()

    def _check_cache(self, query: str) -> dict | None:
        """Check semantic cache for similar query. Returns cached result or None."""
        try:
            cache = self._get_cache()
            if not cache:
                return None
            q_emb = self._embed_query(query)
            best_key = None
            best_sim = 0.0
            for key, entry in cache.items():
                sim = _cosine_sim(q_emb, entry.get("_emb", []))
                if sim > best_sim:
                    best_sim = sim
                    best_key = key
            if best_sim >= _CACHE_SIM_THRESHOLD:
                entry = cache[best_key]
                print(f"[Cache] HIT (sim={best_sim:.3f}) key='{best_key[:80]}...'")
                result = {k: v for k, v in entry.items() if not k.startswith("_")}
                return result
        except Exception as exc:
            print(f"[Cache] Check failed: {exc}")
        return None

    def _store_cache(self, query: str, result: dict) -> None:
        """Store query result in semantic cache."""
        try:
            q_emb = self._embed_query(query)
            entry = {k: v for k, v in result.items() if not k.startswith("_")}
            entry["_emb"] = q_emb
            entry["_ts"] = time.time()
            cache = self._get_cache()
            cache[query] = entry
            if len(cache) % 50 == 0:
                self._flush_cache()
                print(f"[Cache] Auto-flush ({len(cache)} entries)")
        except Exception as exc:
            print(f"[Cache] Store failed: {exc}")

    @classmethod
    def reset_singleton(cls) -> None:
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
            print(f"[QAPipeline] Loading document catalog from {documents_path}")
            with documents_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        key = str(row.get("doc_id") or "").strip()
                        title = str(row.get("doc_title") or "").strip()
                        if key:
                            catalog[key] = row
                        if title and key:
                            title_catalog[title.lower()] = row
                    except Exception:
                        pass
            if catalog:
                break
        print(f"[QAPipeline] Loaded {len(catalog)} documents, {len(title_catalog)} titles")
        return catalog, title_catalog

    def _context_metadata(self, context: dict) -> dict:
        return dict(context.get("metadata") or {})

    def _context_quality(self, query: str, contexts: list[dict], *, route: str) -> dict:
        if not contexts:
            return {"is_relevant": False, "reason": "no_contexts", "score": 0.0}
        best_score = float(contexts[0].get("final_score") or contexts[0].get("score") or 0.0)
        return {
            "is_relevant": best_score > 0.0,
            "reason": "ok" if best_score > 0.0 else "low_score",
            "score": best_score,
        }

    def _build_citation_payload(self, contexts: list[dict]) -> list[dict]:
        if not contexts:
            return []
        citations: list[dict] = []
        seen: set[tuple] = set()
        for context in contexts:
            metadata = self._context_metadata(context)
            score = float(context.get("final_score") or context.get("score") or 0.0)
            key = (metadata.get("doc_title"), metadata.get("article"))
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "doc_title": str(metadata.get("doc_title") or "").strip(),
                "doc_number": str(metadata.get("doc_number") or "").strip(),
                "article": str(metadata.get("article") or "").strip(),
                "citation": str(metadata.get("citation") or "").strip(),
                "source_url": str(metadata.get("source_url") or "").strip(),
                "score": round(score, 4),
            })
        return citations

    def _build_relevant_docs(self, contexts: list[dict]) -> list[str]:
        seen: set[str] = set()
        docs: list[str] = []
        for context in contexts:
            metadata = self._context_metadata(context)
            dn = str(metadata.get("doc_number") or context.get("doc_number") or "").strip()
            dt = str(metadata.get("doc_title") or context.get("doc_title") or "").strip()
            key = dn or dt
            if not key or key in seen:
                continue
            seen.add(key)
            docs.append(f"{dn}|{dt}" if dn and dt else (dn or dt))
        return docs

    def _build_relevant_doc_details(self, contexts: list[dict]) -> list[dict]:
        seen: set[str] = set()
        details: list[dict] = []
        for context in contexts:
            metadata = self._context_metadata(context)
            key = str(metadata.get("doc_number") or metadata.get("doc_id") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            details.append({
                "doc_id": str(metadata.get("doc_id") or "").strip(),
                "doc_number": str(metadata.get("doc_number") or "").strip(),
                "doc_title": str(metadata.get("doc_title") or "").strip(),
                "citation": str(metadata.get("citation") or "").strip(),
                "source_url": str(metadata.get("source_url") or "").strip(),
            })
        return details

    def _build_relevant_articles(self, contexts: list[dict]) -> list[str]:
        seen: set[str] = set()
        articles: list[str] = []
        for context in contexts:
            metadata = self._context_metadata(context)
            dn = str(metadata.get("doc_number") or context.get("doc_number") or "").strip()
            dt = str(metadata.get("doc_title") or context.get("doc_title") or "").strip()
            art = str(metadata.get("article") or context.get("article") or "").strip()
            if not art:
                continue
            key = f"{dn}|{dt}|{art}"
            if key in seen:
                continue
            seen.add(key)
            articles.append(f"{dn}|{dt}|{art}")
        return articles

    def _build_relevant_article_details(self, contexts: list[dict]) -> list[dict]:
        seen: set[str] = set()
        article_details: list[dict] = []
        for context in contexts:
            metadata = self._context_metadata(context)
            doc_title = str(metadata.get("doc_title") or "").strip()
            doc_id = str(metadata.get("doc_id") or "").strip()
            article = str(metadata.get("article") or context.get("article") or "").strip()
            clause = str(metadata.get("clause") or context.get("clause") or "").strip()
            citation = str(metadata.get("citation") or "").strip()
            source_url = str(metadata.get("source_url") or "").strip()
            if not article:
                continue
            key = f"{doc_id}|{article}"
            if key in seen:
                continue
            seen.add(key)
            article_details.append({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "article": article,
                "clause": clause or None,
                "citation": citation,
                "source_url": source_url,
            })
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
        if not contexts:
            return answer_text
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
        if len(answer_text) > 300:
            answer_lower = answer_text.lower()
            for phrase in self._FORBIDDEN_PHRASES:
                if phrase in answer_lower:
                    print(f"[QAPipeline] Answer contained forbidden phrase '{phrase}' but context exists. Regenerating fallback...")
                    citations = self._build_citation_payload(contexts)
                    return self.answer_generator._fallback_answer(contexts=contexts, citations=citations)
            return answer_text
        has_forbidden = False
        for phrase in self._FORBIDDEN_PHRASES:
            if phrase in answer_text.lower():
                has_forbidden = True
                break
        if not has_forbidden:
            for pattern in self._FORBIDDEN_PATTERNS:
                if pattern.search(answer_text):
                    has_forbidden = True
                    break
        if has_forbidden:
            citations = self._build_citation_payload(contexts)
            return self.answer_generator._fallback_answer(contexts=contexts, citations=citations)
        return answer_text

    def _extract_citations_from_answer(self, answer_text: str) -> list[dict]:
        citations = []
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
        # Task 2: Check semantic cache first
        use_cache = os.getenv("R2AI_USE_SEMANTIC_CACHE", "true").strip().lower() in {"1", "true", "yes"}
        if use_cache:
            cached = self._check_cache(question)
            if cached is not None:
                return cached

        t_start = time.perf_counter()

        retrieval_result = self.retrieval_pipeline.run(question)
        raw_final_contexts = list(retrieval_result.get("final_contexts") or [])
        quality = self._context_quality(question, raw_final_contexts, route=str(retrieval_result.get("route") or ""))
        final_contexts = raw_final_contexts if quality["is_relevant"] else []
        answer_retrieval_result = dict(retrieval_result)
        answer_retrieval_result["final_contexts"] = final_contexts

        disable_answer = os.getenv("R2AI_DISABLE_ANSWER", "").strip().lower() in {"1", "true", "yes"}
        if disable_answer:
            result = {
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
            if use_cache:
                self._store_cache(question, result)
            return result

        # Task 3: Generate with structured output (JSON mode via prompt)
        generated = self.answer_generator.generate(query=question, retrieval_result=answer_retrieval_result, use_llm=use_llm)
        citations = self._build_citation_payload(final_contexts)
        relevant_docs = self._build_relevant_docs(final_contexts)
        relevant_doc_details = self._build_relevant_doc_details(final_contexts)
        relevant_articles = self._build_relevant_articles(final_contexts)
        relevant_article_details = self._build_relevant_article_details(final_contexts)

        answer_text = str(generated.get("answer") or "")

        if final_contexts:
            answer_text = self._validate_answer(answer_text, final_contexts)

        if final_contexts and (
            "Chưa đủ căn cứ pháp lý" in answer_text
            or "ChÆ°a Ä‘á»§ cÄƒn cá»© phÃ¡p lÃ½" in answer_text
        ):
            answer_text = self.answer_generator._fallback_answer(contexts=final_contexts, citations=citations)

        answer_citations = self._extract_citations_from_answer(answer_text)

        grounding = None
        if include_grounding:
            grounding = self.grounding_validator.validate(
                query=question,
                answer=answer_text,
                citations=citations,
                contexts=final_contexts,
            )

        elapsed = time.perf_counter() - t_start

        result = {
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

        # Cache result
        if use_cache:
            self._store_cache(question, result)

        # Task 9: Structured monitoring log (JSON)
        print(json.dumps({
            "event": "qa_pipeline",
            "question": question[:100],
            "route": result["route"],
            "elapsed_s": round(elapsed, 2),
            "n_contexts": len(final_contexts),
            "n_docs": len(relevant_docs),
            "n_articles": len(relevant_articles),
            "low_confidence": result["low_confidence"],
            "gen_mode": generated.get("generation_mode", "unknown"),
        }, ensure_ascii=False))

        return result


def _cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run full legal QA pipeline for one question.")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    print(json.dumps(LegalQAPipeline().answer(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
