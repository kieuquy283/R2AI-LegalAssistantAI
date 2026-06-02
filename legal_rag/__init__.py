from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)

SRC_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "legal_rag"
if SRC_PACKAGE_DIR.exists():
    src_package_str = str(SRC_PACKAGE_DIR)
    if src_package_str not in __path__:
        __path__.append(src_package_str)

from .retrieval import DenseRetriever, FAISSRetriever, HybridRetriever, SparseRetriever

__all__ = [
    "DenseRetriever",
    "FAISSRetriever",
    "HybridRetriever",
    "SparseRetriever",
]
