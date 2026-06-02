from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

from langchain_core.documents import Document

from rag.config.retrieval import SHOW_REWRITTEN_QUERY, TOP_K
from rag.generation.answering import answer_with_context_policy
from rag.modules.history_selection import RecencyHistorySelector
from rag.modules.query_rewriting import LLMQueryRewrite
from rag.modules.reranking import NoReranker
from rag.modules.retrieval import FAISSRetriever
from rag.retrieval.ranking import build_top_files


AnswerFn = Callable[[str, str, List[Document], List[Dict[str, Any]]], Dict[str, Any]]


class ModularChatPipeline:
    """
    Modular chat pipeline that composes history selection, query rewriting,
    retrieval, and reranking before delegating answer generation.

    This pipeline is additive and does not replace the legacy ChatPipeline.
    """

    def __init__(
        self,
        index_dir: str = "indexes/default",
        *,
        vectorstore: Any | None = None,
        history_selector: Any | None = None,
        query_rewriter: Any | None = None,
        retriever: Any | None = None,
        reranker: Any | None = None,
        answer_fn: AnswerFn = answer_with_context_policy,
        top_k: int = TOP_K,
        show_rewritten_query: bool = SHOW_REWRITTEN_QUERY,
    ) -> None:
        if vectorstore is None:
            from rag.retrieval.vectorstore import load_vectorstore

            self.vectorstore = load_vectorstore(index_dir=index_dir)
        else:
            self.vectorstore = vectorstore
        self.top_k = int(top_k)
        self.show_rewritten_query = bool(show_rewritten_query)

        self.history_selector = history_selector or RecencyHistorySelector(top_k=3)
        self.query_rewriter = query_rewriter or LLMQueryRewrite()
        self.retriever = retriever or FAISSRetriever(
            vectorstore=self.vectorstore,
            top_k=self.top_k,
        )
        self.reranker = reranker or NoReranker()
        self.answer_fn = answer_fn

    def _results_to_documents(self, results: Sequence[Any]) -> List[Document]:
        docs: List[Document] = []

        for item in results:
            text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue

            metadata = dict(getattr(item, "metadata", {}) or {})
            metadata.setdefault("chunk_id", str(getattr(item, "chunk_id", "unknown")))
            metadata.setdefault("source_file", metadata.get("source", "unknown"))

            raw_score = getattr(item, "raw_score", None)
            if raw_score is not None:
                metadata.setdefault("raw_score", float(raw_score))

            final_score = getattr(item, "final_score", None)
            if final_score is not None:
                metadata.setdefault("final_score", float(final_score))

            score = getattr(item, "score", None)
            if score is not None:
                metadata.setdefault("score", float(score))

            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def chat(
        self,
        question: str,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question rỗng.")

        state: Dict[str, Any] = {
            "question": question,
            "query": question,
            "history": history,
        }

        state = self.history_selector.run(state)
        state = self.query_rewriter.run(state)
        state = self.retriever.run(state)
        state = self.reranker.run(state)

        rewritten_query = str(state.get("rewritten_query") or question).strip()
        used_rewrite = bool(state.get("rewrite_applied", rewritten_query != question.strip()))

        retrieval_results = state.get("reranked_results") or state.get("retrieval_results") or []
        docs = self._results_to_documents(retrieval_results)
        top_files = build_top_files(docs, top_k_files=3)

        answer_payload = self.answer_fn(
            question=question,
            rewritten_query=rewritten_query,
            docs=docs,
            history=history,
        )

        answer = answer_payload["answer"]
        grounded = answer_payload["grounded"]
        warning = answer_payload["warning"]
        mode = answer_payload["mode"]

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "rewritten_query": rewritten_query,
            "used_rewrite": used_rewrite,
            "show_rewritten_query": self.show_rewritten_query,
            "grounded": grounded,
            "warning": warning,
            "mode": mode,
            "top_files": top_files,
            "history": history,
            "pipeline_state": state,
        }

    def chat_stream(
        self,
        question: str,
        history: List[Dict[str, Any]],
    ):
        if not question or not question.strip():
            raise ValueError("Question rỗng.")

        state: Dict[str, Any] = {
            "question": question,
            "query": question,
            "history": history,
        }

        state = self.history_selector.run(state)
        state = self.query_rewriter.run(state)
        state = self.retriever.run(state)
        state = self.reranker.run(state)

        rewritten_query = str(state.get("rewritten_query") or question).strip()
        used_rewrite = bool(state.get("rewrite_applied", rewritten_query != question.strip()))

        retrieval_results = state.get("reranked_results") or state.get("retrieval_results") or []
        docs = self._results_to_documents(retrieval_results)
        top_files = build_top_files(docs, top_k_files=3)

        from rag.generation.answering import stream_answer_with_context_policy
        metadata, stream = stream_answer_with_context_policy(
            question=question,
            rewritten_query=rewritten_query,
            docs=docs,
            history=history,
        )

        full_metadata = {
            "rewritten_query": rewritten_query,
            "used_rewrite": used_rewrite,
            "show_rewritten_query": self.show_rewritten_query,
            "grounded": metadata["grounded"],
            "warning": metadata["warning"],
            "mode": metadata["mode"],
            "top_files": top_files,
        }

        return full_metadata, stream




class LegacyCompatibleModularChatPipeline(ModularChatPipeline):
    """
    Alias-style pipeline for gradual migration from the legacy ChatPipeline.
    """

    pass
