from __future__ import annotations

import os

from rag.config.runtime import get_retrieval_runtime_config
from rag.config.settings import resolve_project_path


TOP_K = int(os.getenv("TOP_K", 3))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", 3))
SHOW_REWRITTEN_QUERY = os.getenv("SHOW_REWRITTEN_QUERY", "true").lower() == "true"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").strip().lower()
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "BAAI/bge-m3",
).strip()

DEFAULT_INDEX_DIR = resolve_project_path(os.getenv("INDEX_DIR", "data/indexes/default"))
DEFAULT_CORPUS_JSON = resolve_project_path("data/retrieval_corpus.json")
DEFAULT_EVALUATION_JSON = resolve_project_path("data/evaluation.json")
DEFAULT_MULTITURN_EVALUATION_JSON = resolve_project_path("data/multiturn_evaluation.json")
DEFAULT_MULTITURN_FILLED_JSON = resolve_project_path("data/multiturn_evaluation_filled.json")

RUNTIME_CONFIG = get_retrieval_runtime_config()
