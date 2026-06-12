from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from src.ingestion.common import read_jsonl, slugify_vi, write_json, write_jsonl


DEFAULT_NODES_PATH = Path("data/processed/legal_nodes.jsonl")
DEFAULT_DOCUMENTS_PATH = Path("data/processed/documents.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_CONTEXT_PATH = Path("data/processed/context_chunks.jsonl")
DEFAULT_EDGES_PATH = Path("data/processed/legal_edges.jsonl")


def _token_count(text: str) -> int:
    return len((text or "").split())


def _build_legal_path(doc_title: str, article: str | None, clause: str | None, point: str | None) -> str:
    parts = [doc_title]
    for value in [article, clause, point]:
        if value:
            parts.append(value)
    return " > ".join(parts)


def _build_citation(doc_title: str, article: str | None, clause: str | None, point: str | None) -> str:
    parts = [doc_title]
    for value in [article, clause, point]:
        if value:
            parts.append(value)
    return ", ".join(parts)


def _build_embedding_text(*parts: object) -> str:
    values = []
    for part in parts:
        text = str(part or "").strip()
        if text:
            values.append(text)
    return "\n".join(values).strip()


def _make_chunk(node: Dict[str, object], document: Dict[str, object], content: str) -> Dict[str, object]:
    chunk_id = slugify_vi(f"{node['node_id']}_{node['level']}")
    doc_title = str(document.get("doc_title") or node.get("doc_id"))
    article = node.get("article")
    clause = node.get("clause")
    point = node.get("point")
    citation = _build_citation(doc_title, article, clause, point)
    legal_path = _build_legal_path(doc_title, article, clause, point)
    return {
        "chunk_id": chunk_id,
        "doc_id": node["doc_id"],
        "node_id": node["node_id"],
        "level": node["level"],
        "domain": node["domain"],
        "doc_title": doc_title,
        "article": article,
        "clause": clause,
        "point": point,
        "legal_path": legal_path,
        "citation": citation,
        "content": content.strip(),
        "embedding_text": _build_embedding_text(
            doc_title,
            node.get("domain"),
            legal_path,
            citation,
            content.strip(),
        ),
        "source_url": node["source_url"],
        "parent_id": node.get("parent_id"),
        "context_chunk_id": None,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "start_char": node.get("start_char"),
    }


def _build_context_chunk(article_node: Dict[str, object], document: Dict[str, object], child_chunk_ids: List[str]) -> Dict[str, object]:
    doc_title = str(document.get("doc_title") or article_node.get("doc_id"))
    article = article_node.get("article")
    context_chunk_id = slugify_vi(f"{article_node['node_id']}_context")
    return {
        "context_chunk_id": context_chunk_id,
        "doc_id": article_node["doc_id"],
        "level": "article",
        "domain": article_node["domain"],
        "doc_title": doc_title,
        "article": article,
        "article_title": article_node.get("article_title"),
        "legal_path": _build_legal_path(doc_title, article, None, None),
        "citation": _build_citation(doc_title, article, None, None),
        "content": str(article_node.get("content") or "").strip(),
        "embedding_text": _build_embedding_text(
            doc_title,
            article_node.get("domain"),
            _build_legal_path(doc_title, article, None, None),
            _build_citation(doc_title, article, None, None),
            str(article_node.get("content") or "").strip(),
        ),
        "source_url": article_node["source_url"],
        "child_chunk_ids": child_chunk_ids,
        "parent_id": article_node.get("parent_id"),
    }


def build_chunks(
    nodes: List[Dict[str, object]],
    documents: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    document_map = {row["doc_id"]: row for row in documents}
    nodes_by_doc: Dict[str, List[Dict[str, object]]] = {}
    for node in nodes:
        nodes_by_doc.setdefault(str(node["doc_id"]), []).append(node)

    chunks: List[Dict[str, object]] = []
    context_chunks: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []

    for doc_id, doc_nodes in nodes_by_doc.items():
        doc_nodes.sort(key=lambda item: int(item.get("start_char", 0)))
        document = document_map[doc_id]
        article_children: Dict[str, List[Dict[str, object]]] = {}

        for node in doc_nodes:
            level = str(node["level"])
            content = str(node.get("content") or node.get("title") or "").strip()
            if not content:
                continue

            if level == "point":
                chunk = _make_chunk(node, document, content)
                chunks.append(chunk)
                article_children.setdefault(str(node["article"]), []).append(chunk)
            elif level == "clause":
                if _token_count(content) <= 1200 and not any(n.get("parent_id") == node["node_id"] and n.get("level") == "point" for n in doc_nodes):
                    chunk = _make_chunk(node, document, content)
                    chunks.append(chunk)
                    article_children.setdefault(str(node["article"]), []).append(chunk)
            elif level == "article":
                child_clauses = [n for n in doc_nodes if n.get("parent_id") == node["node_id"] and n.get("level") in {"clause", "point"}]
                if not child_clauses and _token_count(content) > 0:
                    chunk = _make_chunk(node, document, content)
                    chunks.append(chunk)
                    article_children.setdefault(str(node["article"]), []).append(chunk)

        article_nodes = [node for node in doc_nodes if node.get("level") == "article"]
        for article_node in article_nodes:
            article_key = str(article_node["article"])
            child_chunks = article_children.get(article_key, [])
            if not child_chunks:
                chunk = _make_chunk(article_node, document, str(article_node.get("content") or article_node.get("title") or ""))
                chunks.append(chunk)
                child_chunks = [chunk]
            child_ids = [str(chunk["chunk_id"]) for chunk in child_chunks]
            context_chunk = _build_context_chunk(article_node, document, child_ids)
            context_chunks.append(context_chunk)
            for chunk in child_chunks:
                chunk["context_chunk_id"] = context_chunk["context_chunk_id"]
                chunk["parent_id"] = str(article_node["node_id"]) if chunk["level"] != "article" else article_node.get("parent_id")
                edges.append(
                    {
                        "source_id": chunk["chunk_id"],
                        "target_id": chunk["parent_id"] or context_chunk["context_chunk_id"],
                        "relation_type": "HAS_PARENT",
                        "confidence": 1.0,
                    }
                )

        doc_chunks = [chunk for chunk in chunks if chunk["doc_id"] == doc_id]
        doc_chunks.sort(key=lambda item: (int(item.get("start_char") or 0), str(item["chunk_id"])))
        for index, chunk in enumerate(doc_chunks):
            if index > 0:
                chunk["prev_chunk_id"] = doc_chunks[index - 1]["chunk_id"]
            if index + 1 < len(doc_chunks):
                chunk["next_chunk_id"] = doc_chunks[index + 1]["chunk_id"]
            if chunk.get("prev_chunk_id"):
                edges.append(
                    {
                        "source_id": chunk["chunk_id"],
                        "target_id": chunk["prev_chunk_id"],
                        "relation_type": "PREV_CHUNK",
                        "confidence": 1.0,
                    }
                )
            if chunk.get("next_chunk_id"):
                edges.append(
                    {
                        "source_id": chunk["chunk_id"],
                        "target_id": chunk["next_chunk_id"],
                        "relation_type": "NEXT_CHUNK",
                        "confidence": 1.0,
                    }
                )

    return chunks, context_chunks, edges


def run_legal_chunker(
    *,
    nodes_path: str | Path = DEFAULT_NODES_PATH,
    documents_path: str | Path = DEFAULT_DOCUMENTS_PATH,
    output_path: str | Path = DEFAULT_CHUNKS_PATH,
    context_output_path: str | Path = DEFAULT_CONTEXT_PATH,
    edges_output_path: str | Path = DEFAULT_EDGES_PATH,
) -> Tuple[int, int]:
    nodes = read_jsonl(nodes_path)
    documents = read_jsonl(documents_path)
    chunks, context_chunks, edges = build_chunks(nodes, documents)
    write_jsonl(output_path, chunks)
    write_jsonl(context_output_path, context_chunks)
    write_jsonl(edges_output_path, edges)
    return len(chunks), len(context_chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create retrieval and context chunks from legal nodes.")
    parser.add_argument("--nodes", default=str(DEFAULT_NODES_PATH))
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS_PATH))
    parser.add_argument("--output", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--context-output", default=str(DEFAULT_CONTEXT_PATH))
    parser.add_argument("--edges-output", default=str(DEFAULT_EDGES_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_count, context_count = run_legal_chunker(
        nodes_path=args.nodes,
        documents_path=args.documents,
        output_path=args.output,
        context_output_path=args.context_output,
        edges_output_path=args.edges_output,
    )
    write_json(
        "data/processed/ingestion_report.json",
        {
            "chunks": chunk_count,
            "context_chunks": context_count,
            "chunks_path": args.output,
            "context_chunks_path": args.context_output,
            "edges_path": args.edges_output,
        },
    )
    print(f"Legal chunking: DONE ({chunk_count} chunks, {context_count} context chunks)")


if __name__ == "__main__":
    main()
