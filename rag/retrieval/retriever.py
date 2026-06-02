"""
Legacy compatibility module.

This file is kept to avoid breaking older pipeline/API/evaluation code.
New research retrieval code should use rag.modules.retrieval.
"""

from __future__ import annotations

from typing import Any, List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from rag.config.retrieval import TOP_K


def retrieve_documents(
    query: str,
    vectorstore: FAISS,
    top_k: int | None = None,
    fetch_multiplier: int = 4,
) -> List[Document]:
    """
    Retrieve top-k documents từ vectorstore theo query, kèm raw_score trong metadata.

    Lưu ý:
    - Với FAISS similarity_search_with_score, score thường là distance.
    - Distance càng thấp thì càng gần.
    - Vì index có thể còn inactive chunks, nên over-fetch trước rồi mới filter ở bước sau.
    """
    if not query or not query.strip():
        raise ValueError("Query rỗng, không thể retrieve.")

    k = top_k or TOP_K
    fetch_k = max(k, k * fetch_multiplier)

    docs_and_scores = vectorstore.similarity_search_with_score(query, k=fetch_k)

    results: List[Document] = []
    for doc, score in docs_and_scores:
        metadata = dict(doc.metadata or {})
        metadata["raw_score"] = float(score)

        if "source_file" not in metadata:
            metadata["source_file"] = metadata.get("source", "unknown")

        results.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    # Score nhỏ hơn là tốt hơn vì đây là distance
    results.sort(key=lambda d: float(d.metadata.get("raw_score", 1e9)))
    return results


def extract_cids_from_docs(docs: List[Document]) -> List[Any]:
    cids: List[Any] = []
    for doc in docs:
        cid = doc.metadata.get("cid")
        if cid is not None:
            cids.append(cid)
    return cids
