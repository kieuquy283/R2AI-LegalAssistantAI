"""
Legacy compatibility module.

This file is kept to avoid breaking older pipeline/API/evaluation code.
New research retrieval code should use rag.modules.retrieval and rag.modules.reranking.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from langchain_core.documents import Document


def filter_active_docs(
    docs: List[Document],
    top_k: int | None = None,
) -> List[Document]:
    filtered = [doc for doc in docs if doc.metadata.get("is_active", True)]
    if top_k is not None:
        return filtered[:top_k]
    return filtered


def build_top_files(
    docs: List[Document],
    top_k_files: int = 3,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = defaultdict(list)

    for doc in docs:
        source_file = doc.metadata.get("source_file", "unknown")
        raw_score = float(doc.metadata.get("raw_score", 1e9))
        grouped[source_file].append(raw_score)

    results: List[Dict[str, Any]] = []
    for source_file, scores in grouped.items():
        results.append(
            {
                "source_file": source_file,
                "best_score": min(scores),
                "avg_score": sum(scores) / len(scores),
                "hits": len(scores),
            }
        )

    results.sort(key=lambda x: (x["best_score"], -x["hits"]))
    return results[:top_k_files]
