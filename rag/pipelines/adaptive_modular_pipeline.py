from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain_core.documents import Document

from rag.config.retrieval import SHOW_REWRITTEN_QUERY, TOP_K
from rag.generation.answering import answer_with_context_policy
from rag.modules.history_selection import HybridHistorySelector, RecencyHistorySelector
from rag.modules.query_rewriting import (
    HyDEQueryGenerator,
    LLMQueryRewrite,
    MultiQueryGenerator,
    generate_hyde_query,
)
from rag.modules.reranking import CrossEncoderReranker, NoReranker
from rag.modules.retrieval import FAISSRetriever, HybridRetriever
from rag.modules.routing import AdaptiveRouter, PipelineLevel, compute_retrieval_confidence
from rag.retrieval.ranking import build_top_files
from rag.retrieval.vectorstore import get_embeddings, load_vectorstore
from rag.utils.io import load_json


class AdaptiveModularPipeline:
    def __init__(
        self,
        index_dir: str = "indexes/default",
        *,
        corpus_path: str | None = None,
        vectorstore: Any | None = None,
        answer_fn=answer_with_context_policy,
        top_k: int = TOP_K,
        candidate_k: int = 40,
        history_top_k: int = 4,
        max_history_turns: int = 8,
        show_rewritten_query: bool = SHOW_REWRITTEN_QUERY,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        fusion_mode: str = "rrf",
        reranker_model: str = "BAAI/bge-reranker-base",
        include_original_query: bool = True,
    ) -> None:
        self.index_dir = index_dir
        self.vectorstore = vectorstore
        self.answer_fn = answer_fn
        self.top_k = int(top_k)
        self.candidate_k = int(candidate_k)
        self.history_top_k = int(history_top_k)
        self.max_history_turns = int(max_history_turns)
        self.show_rewritten_query = bool(show_rewritten_query)
        self.dense_weight = float(dense_weight)
        self.sparse_weight = float(sparse_weight)
        self.fusion_mode = fusion_mode
        self.reranker_model = reranker_model
        self.include_original_query = bool(include_original_query)

        self.router = AdaptiveRouter()
        self.corpus_path = corpus_path or self._resolve_default_corpus_path()

        self._embedding_model = None
        self._corpus_documents: List[Document] | None = None
        self._dense_retriever: FAISSRetriever | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._recency_history_selector: RecencyHistorySelector | None = None
        self._hybrid_history_selector: HybridHistorySelector | None = None
        self._query_rewriter: LLMQueryRewrite | None = None
        self._multi_query_generator: MultiQueryGenerator | None = None
        self._hyde_generator: HyDEQueryGenerator | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._no_reranker = NoReranker()

    def _get_vectorstore(self):
        if self.vectorstore is None:
            self.vectorstore = load_vectorstore(index_dir=self.index_dir)
        return self.vectorstore

    def _resolve_default_corpus_path(self) -> str | None:
        candidates = [
            Path("data/legal_corpus_chunks.json"),
            Path("data/retrieval_corpus.json"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _load_corpus_documents(self) -> List[Document]:
        if self._corpus_documents is not None:
            return self._corpus_documents
        if not self.corpus_path:
            self._corpus_documents = []
            return self._corpus_documents

        corpus = load_json(self.corpus_path, [])
        documents: List[Document] = []
        if isinstance(corpus, list):
            for index, item in enumerate(corpus, start=1):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("content") or item.get("text") or "").strip()
                if not text:
                    continue
                item_metadata = dict(item.get("metadata", {}) or {})
                chunk_id = str(item.get("chunk_id") or item_metadata.get("chunk_id") or index)
                cid = str(item.get("cid") or item_metadata.get("cid") or chunk_id)
                metadata = dict(item_metadata)
                metadata["chunk_id"] = chunk_id
                metadata["cid"] = cid
                documents.append(Document(page_content=text, metadata=metadata))
        self._corpus_documents = documents
        return self._corpus_documents

    def _get_embeddings(self):
        if self._embedding_model is None:
            self._embedding_model = get_embeddings()
        return self._embedding_model

    def _get_dense_retriever(self) -> FAISSRetriever:
        if self._dense_retriever is None:
            self._dense_retriever = FAISSRetriever(
                vectorstore=self._get_vectorstore(),
                top_k=self.top_k,
                candidate_k=self.candidate_k,
                filter_active=True,
            )
        return self._dense_retriever

    def _get_hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid_retriever is None:
            total_weight = self.dense_weight + self.sparse_weight
            alpha = 0.5 if total_weight <= 0 else (self.dense_weight / total_weight)
            self._hybrid_retriever = HybridRetriever(
                vectorstore=self._get_vectorstore(),
                documents=self._load_corpus_documents(),
                top_k=self.top_k,
                candidate_k=self.candidate_k,
                fusion_type=self.fusion_mode,
                alpha=alpha,
                filter_active=True,
            )
        return self._hybrid_retriever

    def _get_recency_history_selector(self) -> RecencyHistorySelector:
        if self._recency_history_selector is None:
            self._recency_history_selector = RecencyHistorySelector(top_k=self.history_top_k)
        return self._recency_history_selector

    def _get_hybrid_history_selector(self) -> HybridHistorySelector:
        if self._hybrid_history_selector is None:
            self._hybrid_history_selector = HybridHistorySelector(
                embedding_model=self._get_embeddings(),
                top_k=self.history_top_k,
                alpha=0.8,
                beta=0.2,
                recent_window=self.max_history_turns,
            )
        return self._hybrid_history_selector

    def _get_query_rewriter(self) -> LLMQueryRewrite:
        if self._query_rewriter is None:
            self._query_rewriter = LLMQueryRewrite()
        return self._query_rewriter

    def _get_multi_query_generator(self) -> MultiQueryGenerator:
        if self._multi_query_generator is None:
            self._multi_query_generator = MultiQueryGenerator(num_queries=4)
        return self._multi_query_generator

    def _get_hyde_generator(self) -> HyDEQueryGenerator:
        if self._hyde_generator is None:
            self._hyde_generator = HyDEQueryGenerator()
        return self._hyde_generator

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_name=self.reranker_model,
                candidate_top_k=self.candidate_k,
                output_top_k=self.top_k,
            )
        return self._reranker

    def _results_to_documents(self, results: Sequence[Any]) -> List[Document]:
        docs: List[Document] = []
        for item in results:
            text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue

            metadata = dict(getattr(item, "metadata", {}) or {})
            metadata.setdefault("chunk_id", str(getattr(item, "chunk_id", "unknown")))
            metadata.setdefault("source_file", metadata.get("source", "unknown"))

            for field_name in (
                "raw_score",
                "final_score",
                "score",
                "retrieval_score",
                "rerank_score",
            ):
                value = getattr(item, field_name, None)
                if value is None:
                    continue
                try:
                    metadata.setdefault(field_name, float(value))
                except Exception:
                    continue

            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def _append_unique_query(self, target: List[str], query: str) -> None:
        normalized = str(query or "").strip()
        if not normalized:
            return
        lowered = {item.lower() for item in target}
        if normalized.lower() in lowered:
            return
        target.append(normalized)

    def _final_results_from_state(self, state: Dict[str, Any]) -> List[Any]:
        return list(state.get("reranked_results") or state.get("retrieval_results") or [])

    def _extract_cids(self, results: Sequence[Any]) -> List[str]:
        cids: List[str] = []
        seen = set()
        for item in results:
            metadata = dict(getattr(item, "metadata", {}) or {})
            cid = str(metadata.get("cid") or metadata.get("chunk_id") or getattr(item, "chunk_id", "") or "").strip()
            if cid and cid not in seen:
                seen.add(cid)
                cids.append(cid)
        return cids

    def _run_level(
        self,
        *,
        level: PipelineLevel,
        question: str,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "question": question,
            "query": question,
            "history": list(history),
        }
        route_started_at = time.perf_counter()
        level_metadata: Dict[str, Any] = {
            "selected_history_strategy": "none",
            "used_rewrite": False,
            "used_hybrid": False,
            "used_multi_query": False,
            "used_rerank": False,
            "used_hyde": False,
            "llm_calls": 0,
        }

        if level == PipelineLevel.REWRITE_DENSE:
            state = self._get_recency_history_selector().run(state)
            level_metadata["selected_history_strategy"] = "recency"
            state = self._get_query_rewriter().run(state)
            level_metadata["used_rewrite"] = bool(state.get("rewrite_applied"))
            level_metadata["llm_calls"] += 1 if state.get("selected_history") else 0
            state = self._get_dense_retriever().run(state)
            retrieval_results = list(state.get("retrieval_results", []) or [])
            state["reranked_results"] = retrieval_results
            state["reranking"] = self._no_reranker.build_metadata(
                strategy="no_reranker",
                rerank_applied=False,
                input_count=len(retrieval_results),
                output_count=len(retrieval_results),
                model_name=None,
            )
        elif level == PipelineLevel.HYBRID:
            state = self._get_hybrid_retriever().run(state)
            level_metadata["used_hybrid"] = True
            retrieval_results = list(state.get("retrieval_results", []) or [])
            state["reranked_results"] = retrieval_results
            state["reranking"] = self._no_reranker.build_metadata(
                strategy="no_reranker",
                rerank_applied=False,
                input_count=len(retrieval_results),
                output_count=len(retrieval_results),
                model_name=None,
            )
        elif level == PipelineLevel.HYBRID_RERANK:
            state = self._get_hybrid_retriever().run(state)
            level_metadata["used_hybrid"] = True
            state = self._get_reranker().run(state)
            level_metadata["used_rerank"] = bool(state.get("retrieval_results"))
        elif level == PipelineLevel.FULL_OPTIMAL:
            state = self._get_hybrid_history_selector().run(state)
            level_metadata["selected_history_strategy"] = "hybrid"
            state = self._get_query_rewriter().run(state)
            level_metadata["used_rewrite"] = bool(state.get("rewrite_applied"))
            level_metadata["llm_calls"] += 1
            state = self._get_multi_query_generator().run(state)
            level_metadata["used_multi_query"] = len(state.get("queries", []) or []) > 1
            level_metadata["llm_calls"] += 1
            state = self._get_hybrid_retriever().run(state)
            level_metadata["used_hybrid"] = True
            state = self._get_reranker().run(state)
            level_metadata["used_rerank"] = bool(state.get("retrieval_results"))
        elif level == PipelineLevel.HYDE:
            state = self._get_hybrid_history_selector().run(state)
            level_metadata["selected_history_strategy"] = "hybrid"
            state = self._get_query_rewriter().run(state)
            level_metadata["used_rewrite"] = bool(state.get("rewrite_applied"))
            level_metadata["llm_calls"] += 1

            rewritten_query = str(state.get("rewritten_query") or question).strip()
            selected_history = list(state.get("selected_history", []) or [])
            hyde_query, hyde_used, hyde_failed = generate_hyde_query(
                rewritten_query=rewritten_query,
                selected_history=selected_history,
                mode="llm",
                generator=self._get_hyde_generator(),
            )
            queries: List[str] = []
            if self.include_original_query:
                self._append_unique_query(queries, question)
            self._append_unique_query(queries, rewritten_query)
            if hyde_used:
                self._append_unique_query(queries, hyde_query)

            state["hyde_query"] = hyde_query
            state["hyde_used"] = hyde_used
            state["hyde_failed"] = hyde_failed
            state["hyde"] = {
                "strategy": "llm_hyde",
                "input_query": rewritten_query,
                "hyde_query": hyde_query,
                "hyde_used": hyde_used,
                "hyde_failed": hyde_failed,
            }
            state["queries"] = queries or [rewritten_query or question]
            level_metadata["used_hyde"] = bool(hyde_used)
            level_metadata["used_hybrid"] = True
            level_metadata["llm_calls"] += 1

            state = self._get_hybrid_retriever().run(state)
            state = self._get_reranker().run(state)
            level_metadata["used_rerank"] = bool(state.get("retrieval_results"))
        else:
            state = self._get_dense_retriever().run(state)
            retrieval_results = list(state.get("retrieval_results", []) or [])
            state["reranked_results"] = retrieval_results
            state["reranking"] = self._no_reranker.build_metadata(
                strategy="no_reranker",
                rerank_applied=False,
                input_count=len(retrieval_results),
                output_count=len(retrieval_results),
                model_name=None,
            )

        state["route_level"] = level.value
        state["route_metadata"] = level_metadata
        state["route_latency_seconds"] = time.perf_counter() - route_started_at
        return state

    def run_retrieval(
        self,
        question: str,
        history: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question rỗng.")

        normalized_history = list(history or [])
        decision = self.router.choose_initial_route(question=question, history=normalized_history)
        selected_route = decision.level
        current_level = selected_route
        escalation_path = [selected_route.value]
        route_runs: List[Dict[str, Any]] = []

        while current_level is not None:
            state = self._run_level(
                level=current_level,
                question=question,
                history=normalized_history,
            )
            final_results = self._final_results_from_state(state)
            confidence = compute_retrieval_confidence(final_results, expected_top_k=self.top_k)

            state["retrieval_confidence"] = {
                "score": confidence.score,
                "doc_count": confidence.doc_count,
                "unique_cid_count": confidence.unique_cid_count,
                "has_cid_metadata": confidence.has_cid_metadata,
                "non_empty_content_ratio": confidence.non_empty_content_ratio,
                "duplicate_rate": confidence.duplicate_rate,
                "score_signal_available": confidence.score_signal_available,
                "top_score": confidence.top_score,
                "score_gap": confidence.score_gap,
                "reasons": confidence.reasons,
            }
            route_runs.append(state)

            if not self.router.should_escalate(confidence=confidence, current_level=current_level):
                break

            next_level = self.router.compute_next_level(current_level)
            if next_level is None:
                break
            current_level = next_level
            escalation_path.append(current_level.value)

        final_state = route_runs[-1]
        final_results = self._final_results_from_state(final_state)
        final_confidence = final_state["retrieval_confidence"]
        final_query = str(
            final_state.get("hyde_query")
            if final_state.get("hyde_used")
            else final_state.get("rewritten_query")
            or final_state.get("query")
            or question
        ).strip()

        final_state["adaptive_metadata"] = {
            "pipeline_mode": "adaptive",
            "selected_route": selected_route.value,
            "final_route": final_state["route_level"],
            "escalation_path": escalation_path,
            "escalated": len(escalation_path) > 1,
            "confidence_score": final_confidence["score"],
            "used_rewrite": any(bool(run.get("route_metadata", {}).get("used_rewrite")) for run in route_runs),
            "used_hybrid": any(bool(run.get("route_metadata", {}).get("used_hybrid")) for run in route_runs),
            "used_multi_query": any(bool(run.get("route_metadata", {}).get("used_multi_query")) for run in route_runs),
            "used_rerank": any(bool(run.get("route_metadata", {}).get("used_rerank")) for run in route_runs),
            "used_hyde": any(bool(run.get("route_metadata", {}).get("used_hyde")) for run in route_runs),
            "llm_calls": sum(int(run.get("route_metadata", {}).get("llm_calls", 0)) for run in route_runs),
            "route_latencies": {
                run["route_level"]: float(run.get("route_latency_seconds", 0.0))
                for run in route_runs
            },
            "query_used": final_query or question,
            "retrieved_cids": self._extract_cids(final_results),
            "route_reasons": list(decision.reasons),
            "analyzer_signals": dict(decision.analyzer_signals),
        }
        return final_state

    def chat(
        self,
        question: str,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        runtime_started_at = time.perf_counter()
        input_history = list(history or [])
        state = self.run_retrieval(question=question, history=input_history)
        final_results = self._final_results_from_state(state)
        docs = self._results_to_documents(final_results)
        top_files = build_top_files(docs, top_k_files=3)

        rewritten_query = str(state.get("rewritten_query") or question).strip()
        answer_payload = self.answer_fn(
            question=question,
            rewritten_query=rewritten_query,
            docs=docs,
            history=input_history,
        )

        answer = answer_payload["answer"]
        updated_history = list(input_history)
        updated_history.append({"role": "user", "content": question})
        updated_history.append({"role": "assistant", "content": answer})

        metadata = dict(state.get("adaptive_metadata", {}) or {})
        metadata["latency_seconds"] = time.perf_counter() - runtime_started_at

        return {
            "answer": answer,
            "rewritten_query": rewritten_query,
            "used_rewrite": bool(metadata.get("used_rewrite")),
            "show_rewritten_query": self.show_rewritten_query,
            "grounded": answer_payload["grounded"],
            "warning": answer_payload["warning"],
            "mode": answer_payload["mode"],
            "top_files": top_files,
            "history": updated_history,
            "metadata": metadata,
            "pipeline_state": state,
        }
