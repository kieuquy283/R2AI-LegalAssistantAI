"""Legal corpus models and helpers."""

from legal_rag.corpus.normalize import load_articles_jsonl, normalize_corpus_items, save_articles_jsonl
from legal_rag.corpus.schema import LegalArticle

__all__ = [
    "LegalArticle",
    "load_articles_jsonl",
    "normalize_corpus_items",
    "save_articles_jsonl",
]
