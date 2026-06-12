from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.ingestion.common import read_jsonl, slugify_vi, write_jsonl


DEFAULT_DOCUMENTS_PATH = Path("data/processed/documents.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_NODES_PATH = Path("data/processed/legal_nodes.jsonl")
DEFAULT_EDGES_PATH = Path("data/processed/legal_edges.jsonl")
DEFAULT_EXPLICIT_REFS_PATH = Path("data/processed/explicit_refs.jsonl")
DEFAULT_CROSS_DOMAIN_EDGES_PATH = Path("data/processed/cross_domain_edges.jsonl")
DEFAULT_DOMAIN_TAXONOMY_PATH = Path("data/sources/domain_taxonomy.json")
DEFAULT_OUTPUT_NODES_PATH = Path("data/processed/legal_graph_nodes.jsonl")
DEFAULT_OUTPUT_EDGES_PATH = Path("data/processed/legal_graph_edges.jsonl")


def _load_taxonomy(path: str | Path) -> Dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _node_row(
    node_id: str,
    node_type: str,
    *,
    doc_id: str | None = None,
    domain: str | None = None,
    title: str | None = None,
    content: str | None = None,
    source_url: str | None = None,
    metadata: Dict[str, object] | None = None,
) -> Dict[str, object]:
    return {
        "node_id": node_id,
        "node_type": node_type,
        "doc_id": doc_id,
        "domain": domain,
        "title": title,
        "content": content,
        "source_url": source_url,
        "metadata": metadata or {},
    }


def _edge_row(
    source_id: str,
    target_id: str,
    relation_type: str,
    *,
    target_domain: str | None = None,
    ref_text: str | None = None,
    confidence: float = 1.0,
    metadata: Dict[str, object] | None = None,
) -> Dict[str, object]:
    edge_id = slugify_vi(f"{source_id}|{relation_type}|{target_id}|{target_domain or ''}|{ref_text or ''}")
    return {
        "edge_id": edge_id,
        "source_id": source_id,
        "target_id": target_id,
        "target_domain": target_domain,
        "relation_type": relation_type,
        "ref_text": ref_text,
        "confidence": float(confidence),
        "metadata": metadata or {},
    }


def build_graph(
    *,
    documents: List[Dict[str, object]],
    chunks: List[Dict[str, object]],
    legal_nodes: List[Dict[str, object]],
    legal_edges: List[Dict[str, object]],
    explicit_refs: List[Dict[str, object]],
    cross_domain_edges: List[Dict[str, object]],
    taxonomy: Dict[str, dict],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    nodes: Dict[str, Dict[str, object]] = {}
    edges: Dict[str, Dict[str, object]] = {}

    def add_node(row: Dict[str, object]) -> None:
        nodes[str(row["node_id"])] = row

    def add_edge(row: Dict[str, object]) -> None:
        edges[str(row["edge_id"])] = row

    for domain, meta in taxonomy.items():
        add_node(
            _node_row(
                f"domain:{domain}",
                "domain",
                domain=domain,
                title=domain,
                content=str(meta.get("description") or ""),
                metadata={"keywords": meta.get("keywords", []), "subdomains": meta.get("subdomains", [])},
            )
        )

    for document in documents:
        doc_id = str(document["doc_id"])
        add_node(
            _node_row(
                doc_id,
                "document",
                doc_id=doc_id,
                domain=str(document.get("domain") or ""),
                title=str(document.get("doc_title") or doc_id),
                content=str(document.get("doc_title") or ""),
                source_url=str(document.get("source_url") or ""),
                metadata={
                    "doc_number": document.get("doc_number"),
                    "doc_type": document.get("doc_type"),
                    "issuing_body": document.get("issuing_body"),
                    "issue_date": document.get("issue_date"),
                    "effective_date": document.get("effective_date"),
                    "status": document.get("status"),
                    "provider": document.get("provider"),
                },
            )
        )
        if document.get("domain"):
            add_edge(
                _edge_row(
                    doc_id,
                    f"domain:{document['domain']}",
                    "SAME_DOMAIN",
                    target_domain=str(document["domain"]),
                    metadata={"source_type": "document"},
                )
            )

    for node in legal_nodes:
        node_id = str(node["node_id"])
        add_node(
            _node_row(
                node_id,
                str(node.get("level") or "article"),
                doc_id=str(node.get("doc_id") or ""),
                domain=str(node.get("domain") or ""),
                title=str(node.get("title") or ""),
                content=str(node.get("content") or ""),
                source_url=str(node.get("source_url") or ""),
                metadata={
                    "article": node.get("article"),
                    "article_title": node.get("article_title"),
                    "clause": node.get("clause"),
                    "point": node.get("point"),
                    "parent_id": node.get("parent_id"),
                    "start_char": node.get("start_char"),
                    "end_char": node.get("end_char"),
                },
            )
        )
        if node.get("domain"):
            add_edge(
                _edge_row(
                    node_id,
                    f"domain:{node['domain']}",
                    "SAME_DOMAIN",
                    target_domain=str(node["domain"]),
                    metadata={"source_type": "legal_node"},
                )
            )
        if node.get("doc_id"):
            add_edge(
                _edge_row(
                    node_id,
                    str(node["doc_id"]),
                    "HAS_PARENT",
                    metadata={"source_type": "legal_node", "parent_type": "document"},
                )
            )
            add_edge(
                _edge_row(
                    str(node["doc_id"]),
                    node_id,
                    "HAS_CHILD",
                    metadata={"source_type": "legal_node", "child_type": str(node.get("level") or "")},
                )
            )

    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        add_node(
            _node_row(
                chunk_id,
                str(chunk.get("level") or "article"),
                doc_id=str(chunk.get("doc_id") or ""),
                domain=str(chunk.get("domain") or ""),
                title=str(chunk.get("citation") or chunk.get("legal_path") or chunk_id),
                content=str(chunk.get("content") or ""),
                source_url=str(chunk.get("source_url") or ""),
                metadata={
                    "citation": chunk.get("citation"),
                    "article": chunk.get("article"),
                    "clause": chunk.get("clause"),
                    "point": chunk.get("point"),
                    "node_id": chunk.get("node_id"),
                    "parent_id": chunk.get("parent_id"),
                    "prev_chunk_id": chunk.get("prev_chunk_id"),
                    "next_chunk_id": chunk.get("next_chunk_id"),
                    "context_chunk_id": chunk.get("context_chunk_id"),
                    "doc_title": chunk.get("doc_title"),
                },
            )
        )
        if chunk.get("domain"):
            add_edge(
                _edge_row(
                    chunk_id,
                    f"domain:{chunk['domain']}",
                    "SAME_DOMAIN",
                    target_domain=str(chunk["domain"]),
                    metadata={"source_type": "chunk"},
                )
            )
        if chunk.get("doc_id"):
            add_edge(
                _edge_row(
                    chunk_id,
                    str(chunk["doc_id"]),
                    "HAS_PARENT",
                    metadata={"source_type": "chunk", "parent_type": "document"},
                )
            )
            add_edge(
                _edge_row(
                    str(chunk["doc_id"]),
                    chunk_id,
                    "HAS_CHILD",
                    metadata={"source_type": "chunk", "child_type": str(chunk.get("level") or "")},
                )
            )

    relation_map = {"PREV_CHUNK": "PREVIOUS", "NEXT_CHUNK": "NEXT"}
    for edge in legal_edges:
        source_id = str(edge.get("source_id") or "")
        target_id = str(edge.get("target_id") or "")
        relation_type = relation_map.get(str(edge.get("relation_type") or ""), str(edge.get("relation_type") or ""))
        if not source_id or not target_id:
            continue
        add_edge(
            _edge_row(
                source_id,
                target_id,
                relation_type,
                confidence=float(edge.get("confidence") or 1.0),
                metadata={"source": "legal_edges"},
            )
        )
        if relation_type == "HAS_PARENT":
            add_edge(
                _edge_row(
                    target_id,
                    source_id,
                    "HAS_CHILD",
                    confidence=float(edge.get("confidence") or 1.0),
                    metadata={"source": "legal_edges"},
                )
            )

    for ref in explicit_refs:
        source_id = str(ref.get("source_chunk_id") or "")
        target_id = str(ref.get("target_chunk_id") or ref.get("target_doc_id") or "")
        if not source_id or not target_id:
            continue
        add_edge(
            _edge_row(
                source_id,
                target_id,
                "REFERS_TO",
                target_domain=str(ref.get("target_domain") or "") or None,
                ref_text=str(ref.get("match_text") or "") or None,
                confidence=float(ref.get("confidence") or 1.0),
                metadata={
                    "source_doc_id": ref.get("source_doc_id"),
                    "target_doc_id": ref.get("target_doc_id"),
                    "target_article": ref.get("target_article"),
                    "target_article_ref": ref.get("target_article_ref"),
                    "ref_type": ref.get("ref_type"),
                    "resolution": ref.get("resolution"),
                    "is_cross_doc": ref.get("is_cross_doc"),
                    "is_cross_domain": ref.get("is_cross_domain"),
                },
            )
        )

    for edge in cross_domain_edges:
        source_id = str(edge.get("source_id") or "")
        target_domain = str(edge.get("target_domain") or "")
        target_id = str(edge.get("target_id") or f"domain:{target_domain}")
        if not source_id or not target_id:
            continue
        add_edge(
            _edge_row(
                source_id,
                target_id,
                "CROSS_DOMAIN",
                target_domain=target_domain or None,
                ref_text=str(edge.get("match_text") or "") or None,
                confidence=float(edge.get("confidence") or 1.0),
                metadata={
                    "source_domain": edge.get("source_domain"),
                    "source_relation_type": edge.get("relation_type"),
                },
            )
        )

    ordered_nodes = sorted(nodes.values(), key=lambda row: (str(row.get("node_type") or ""), str(row["node_id"])))
    ordered_edges = sorted(edges.values(), key=lambda row: (str(row.get("relation_type") or ""), str(row["source_id"]), str(row["target_id"])))
    return ordered_nodes, ordered_edges


def run_graph_builder(
    *,
    documents_path: str | Path = DEFAULT_DOCUMENTS_PATH,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    nodes_path: str | Path = DEFAULT_NODES_PATH,
    edges_path: str | Path = DEFAULT_EDGES_PATH,
    explicit_refs_path: str | Path = DEFAULT_EXPLICIT_REFS_PATH,
    cross_domain_edges_path: str | Path = DEFAULT_CROSS_DOMAIN_EDGES_PATH,
    domain_taxonomy_path: str | Path = DEFAULT_DOMAIN_TAXONOMY_PATH,
    output_nodes_path: str | Path = DEFAULT_OUTPUT_NODES_PATH,
    output_edges_path: str | Path = DEFAULT_OUTPUT_EDGES_PATH,
) -> Tuple[int, int]:
    graph_nodes, graph_edges = build_graph(
        documents=read_jsonl(documents_path),
        chunks=read_jsonl(chunks_path),
        legal_nodes=read_jsonl(nodes_path),
        legal_edges=read_jsonl(edges_path),
        explicit_refs=read_jsonl(explicit_refs_path),
        cross_domain_edges=read_jsonl(cross_domain_edges_path),
        taxonomy=_load_taxonomy(domain_taxonomy_path),
    )
    node_count = write_jsonl(output_nodes_path, graph_nodes)
    edge_count = write_jsonl(output_edges_path, graph_edges)
    return node_count, edge_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical lightweight legal graph files.")
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS_PATH))
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--nodes", default=str(DEFAULT_NODES_PATH))
    parser.add_argument("--edges", default=str(DEFAULT_EDGES_PATH))
    parser.add_argument("--explicit-refs", default=str(DEFAULT_EXPLICIT_REFS_PATH))
    parser.add_argument("--cross-domain-edges", default=str(DEFAULT_CROSS_DOMAIN_EDGES_PATH))
    parser.add_argument("--domain-taxonomy", default=str(DEFAULT_DOMAIN_TAXONOMY_PATH))
    parser.add_argument("--output-nodes", default=str(DEFAULT_OUTPUT_NODES_PATH))
    parser.add_argument("--output-edges", default=str(DEFAULT_OUTPUT_EDGES_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_count, edge_count = run_graph_builder(
        documents_path=args.documents,
        chunks_path=args.chunks,
        nodes_path=args.nodes,
        edges_path=args.edges,
        explicit_refs_path=args.explicit_refs,
        cross_domain_edges_path=args.cross_domain_edges,
        domain_taxonomy_path=args.domain_taxonomy,
        output_nodes_path=args.output_nodes,
        output_edges_path=args.output_edges,
    )
    print(f"Canonical legal graph: DONE ({node_count} nodes, {edge_count} edges)")


if __name__ == "__main__":
    main()
