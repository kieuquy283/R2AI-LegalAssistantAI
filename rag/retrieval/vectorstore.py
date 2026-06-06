from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path
from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from rag.config.retrieval import DEFAULT_INDEX_DIR, EMBEDDING_BACKEND, LOCAL_EMBEDDING_MODEL

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_EMBEDDINGS_CACHE: Embeddings | None = None


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            [f"query: {text}"],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()


class OfflineHashEmbeddings(Embeddings):
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed_one(self, text: str, prefix: str) -> List[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        payload = f"{prefix}:{text}".encode("utf-8")
        tokens = (text or "").split() or [text or "_empty_"]
        for token in tokens:
            digest = hashlib.sha256(payload + token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0)
            vector[index] += sign * weight

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            norm = 1.0
        vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text, "passage") for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text, "query")


def get_embeddings() -> Embeddings:
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE

    if EMBEDDING_BACKEND == "local":
        try:
            _EMBEDDINGS_CACHE = LocalSentenceTransformerEmbeddings(LOCAL_EMBEDDING_MODEL)
        except Exception as exc:
            warnings.warn(
                f"Falling back to offline hash embeddings because '{LOCAL_EMBEDDING_MODEL}' could not be loaded: {exc}",
                RuntimeWarning,
            )
            _EMBEDDINGS_CACHE = OfflineHashEmbeddings()
        return _EMBEDDINGS_CACHE

    raise ValueError("Chi ho tro EMBEDDING_BACKEND=local")


def ensure_index_dir(index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
    Path(index_dir).mkdir(parents=True, exist_ok=True)


def build_and_save_vectorstore(
    documents: List[Document],
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> FAISS:
    if not documents:
        raise ValueError("Khong co documents de build vectorstore.")

    ensure_index_dir(index_dir)
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(index_dir))

    return vectorstore


def load_vectorstore(index_dir: str | Path = DEFAULT_INDEX_DIR) -> FAISS:
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc index: {index_dir}")

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
