from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.config.runtime import get_retrieval_runtime_config
from rag.retrieval.vectorstore import get_embeddings
from src.ingestion.common import read_jsonl, write_json
from src.retrieval.qdrant_store import QdrantStore


def _embedding_text_for_document(row: dict[str, Any]) -> str:
    parts = [
        row.get("doc_title"),
        row.get("doc_number"),
        row.get("doc_type"),
        row.get("issuer"),
        row.get("domain"),
        row.get("cleaned_text"),
    ]
    return "\n".join(str(part or "").strip() for part in parts if str(part or "").strip())


def _embedding_text_for_article(row: dict[str, Any]) -> str:
    parts = [
        row.get("title"),
        row.get("doc_id"),
        row.get("article"),
        row.get("article_title"),
        row.get("clause"),
        row.get("point"),
        row.get("content"),
        row.get("domain"),
    ]
    return "\n".join(str(part or "").strip() for part in parts if str(part or "").strip())


def _embedding_text_for_chunk(row: dict[str, Any]) -> str:
    if str(row.get("embedding_text") or "").strip():
        return str(row["embedding_text"]).strip()
    parts = [
        row.get("doc_title"),
        row.get("article"),
        row.get("clause"),
        row.get("citation"),
        row.get("content"),
        row.get("domain"),
    ]
    return "\n".join(str(part or "").strip() for part in parts if str(part or "").strip())


def _with_vectors(rows: list[dict[str, Any]], text_builder) -> list[dict[str, Any]]:
    embeddings = get_embeddings()
    texts = [text_builder(row) for row in rows]
    vectors = embeddings.embed_documents(texts)
    enriched: list[dict[str, Any]] = []
    for row, text, vector in zip(rows, texts, vectors, strict=True):
        item = dict(row)
        item["source_dataset"] = str(item.get("source_dataset") or "local_corpus")
        item["embedding_text"] = text
        item["vector"] = vector
        enriched.append(item)
    return enriched


def build_qdrant_index(
    *,
    documents_path: str | Path,
    articles_path: str | Path,
    chunks_path: str | Path,
    recreate: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    config = get_retrieval_runtime_config()
    documents = read_jsonl(documents_path)
    articles = read_jsonl(articles_path)
    chunks = read_jsonl(chunks_path)
    if limit is not None:
        documents = documents[:limit]
        articles = articles[:limit]
        chunks = chunks[:limit]

    qdrant = QdrantStore(config=config)
    target_collections = [
        config.qdrant_collection_docs,
        config.qdrant_collection_articles,
        config.qdrant_collection_chunks,
    ]
    for collection in target_collections:
        if recreate:
            qdrant.recreate_collection(collection)
        else:
            qdrant.ensure_collection(collection)

    document_points = _with_vectors(documents, _embedding_text_for_document)
    article_points = _with_vectors(articles, _embedding_text_for_article)
    chunk_points = _with_vectors(chunks, _embedding_text_for_chunk)

    inserted_docs = qdrant.upsert_rows(
        collection_name=config.qdrant_collection_docs,
        rows=document_points,
        vector_key="vector",
        id_key="doc_id",
    )
    inserted_articles = qdrant.upsert_rows(
        collection_name=config.qdrant_collection_articles,
        rows=article_points,
        vector_key="vector",
        id_key="node_id",
    )
    inserted_chunks = qdrant.upsert_rows(
        collection_name=config.qdrant_collection_chunks,
        rows=chunk_points,
        vector_key="vector",
        id_key="chunk_id",
    )

    return {
        "documents_inserted": inserted_docs,
        "articles_inserted": inserted_articles,
        "chunks_inserted": inserted_chunks,
        "legal_docs_count": qdrant.count(config.qdrant_collection_docs),
        "legal_articles_count": qdrant.count(config.qdrant_collection_articles),
        "legal_chunks_count": qdrant.count(config.qdrant_collection_chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant multi-level collections from legal corpus artifacts.")
    parser.add_argument("--documents", required=True)
    parser.add_argument("--articles", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default="logs/debug/qdrant_index_report.json")
    args = parser.parse_args()

    report = build_qdrant_index(
        documents_path=args.documents,
        articles_path=args.articles,
        chunks_path=args.chunks,
        recreate=args.recreate,
        limit=args.limit,
    )
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
