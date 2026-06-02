"""
Legacy compatibility module.

This file is kept to avoid breaking older pipeline/API/evaluation code.
New code should use rag.modules.* composition under a modular pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rag.retrieval.vectorstore import load_vectorstore
from rag.retrieval.retriever import retrieve_documents
from rag.retrieval.ranking import filter_active_docs, build_top_files
from rag.retrieval.query_rewriter import rewrite_query
from rag.generation.answering import answer_with_context_policy
from rag.config.retrieval import TOP_K, SHOW_REWRITTEN_QUERY


class ChatPipeline:
    def __init__(self, index_dir: str = "indexes/default"):
        self.vectorstore = load_vectorstore(index_dir=index_dir)

    def chat(
        self,
        question: str,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question rỗng.")

        # 1) Rewrite query
        if not history:
            rewritten_query = question
            used_rewrite = False
        else:
            rewritten_query = rewrite_query(
                current_question=question,
                history=history,
            )
            used_rewrite = rewritten_query.strip() != question.strip()

        # 2) Retrieve docs
        docs = retrieve_documents(
            query=rewritten_query,
            vectorstore=self.vectorstore,
            top_k=TOP_K,
        )

        # 3) Filter active docs
        docs = filter_active_docs(docs, top_k=TOP_K)

        # 4) Build top files
        top_files = build_top_files(docs, top_k_files=3)

        # 5) Answer with policy
        result = answer_with_context_policy(
            question=question,
            rewritten_query=rewritten_query,
            docs=docs,
            history=history,
        )

        answer = result["answer"]
        grounded = result["grounded"]
        warning = result["warning"]
        mode = result["mode"]

        # 6) Update history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "rewritten_query": rewritten_query,
            "used_rewrite": used_rewrite,
            "show_rewritten_query": SHOW_REWRITTEN_QUERY,
            "grounded": grounded,
            "warning": warning,
            "mode": mode,
            "top_files": top_files,
            "history": history,
        }

    def chat_stream(
        self,
        question: str,
        history: List[Dict[str, Any]],
    ):
        if not question or not question.strip():
            raise ValueError("Question rỗng.")

        # 1) Rewrite query
        if not history:
            rewritten_query = question
            used_rewrite = False
        else:
            rewritten_query = rewrite_query(
                current_question=question,
                history=history,
            )
            used_rewrite = rewritten_query.strip() != question.strip()

        # 2) Retrieve docs
        docs = retrieve_documents(
            query=rewritten_query,
            vectorstore=self.vectorstore,
            top_k=TOP_K,
        )

        # 3) Filter active docs
        docs = filter_active_docs(docs, top_k=TOP_K)

        # 4) Build top files
        top_files = build_top_files(docs, top_k_files=3)

        # 5) Answer with policy stream
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
            "show_rewritten_query": SHOW_REWRITTEN_QUERY,
            "grounded": metadata["grounded"],
            "warning": metadata["warning"],
            "mode": metadata["mode"],
            "top_files": top_files,
        }

        return full_metadata, stream

