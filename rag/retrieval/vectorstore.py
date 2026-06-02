from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from rag.config.retrieval import DEFAULT_INDEX_DIR, EMBEDDING_BACKEND, LOCAL_EMBEDDING_MODEL


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


def get_embeddings() -> Embeddings:
    if EMBEDDING_BACKEND == "local":
        return LocalSentenceTransformerEmbeddings(LOCAL_EMBEDDING_MODEL)

    raise ValueError("Chỉ hỗ trợ EMBEDDING_BACKEND=local")


def ensure_index_dir(index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
    Path(index_dir).mkdir(parents=True, exist_ok=True)


def build_and_save_vectorstore(
    documents: List[Document],
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> FAISS:
    if not documents:
        raise ValueError("Không có documents để build vectorstore.")

    ensure_index_dir(index_dir)
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(index_dir))

    return vectorstore


def load_vectorstore(index_dir: str | Path = DEFAULT_INDEX_DIR) -> FAISS:
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục index: {index_dir}")

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
