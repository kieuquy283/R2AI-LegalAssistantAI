"""Stable package surface for the R2AI legal assistant."""

from legal_rag.retrieval import DenseRetriever, FAISSRetriever, HybridRetriever, SparseRetriever

__all__ = [
    "DenseRetriever",
    "FAISSRetriever",
    "HybridRetriever",
    "SparseRetriever",
]
