from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.ingestion.common import read_jsonl, write_jsonl


DEFAULT_DOCUMENTS_PATH = Path("data/processed/documents.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_CONTEXT_CHUNKS_PATH = Path("data/processed/context_chunks.jsonl")
DEFAULT_TAXONOMY_PATH = Path("data/sources/domain_taxonomy.json")
DEFAULT_EXPLICIT_REFS_PATH = Path("data/processed/explicit_refs.jsonl")
DEFAULT_CROSS_DOMAIN_EDGES_PATH = Path("data/processed/cross_domain_edges.jsonl")

ARTICLE_REF_RE = re.compile(r"Điều\s+\d+[A-Za-z]?", re.IGNORECASE)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").lower()


def _load_taxonomy(path: str | Path) -> Dict[str, dict]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _doc_ref(document: Dict[str, object]) -> str:
    doc_number = str(document.get("doc_number") or document.get("doc_id") or "").strip()
    doc_title = str(document.get("doc_title") or document.get("doc_id") or "").strip()
    return f"{doc_number}|{doc_title}"


def _normalize_article_label(value: str) -> str:
    match = re.search(r"(\d+[A-Za-z]?)", value or "")
    if not match:
        return " ".join(str(value or "").split())
    return f"Điều {match.group(1)}"


def _article_ref(document: Dict[str, object], article: str | None) -> str | None:
    if not article:
        return None
    return f"{_doc_ref(document)}|{_normalize_article_label(article)}"


def _build_article_index(
    chunks: List[Dict[str, object]],
    documents_by_id: Dict[str, Dict[str, object]],
) -> Dict[Tuple[str, str], Dict[str, object]]:
    article_index: Dict[Tuple[str, str], Dict[str, object]] = {}
    for chunk in chunks:
        article = str(chunk.get("article") or "").strip()
        doc_id = str(chunk["doc_id"])
        if not article:
            continue
        key = (doc_id, article)
        current = article_index.get(key)
        if current is None or chunk["level"] == "article":
            article_index[key] = {
                "chunk_id": chunk["chunk_id"],
                "context_chunk_id": chunk.get("context_chunk_id"),
                "doc_id": doc_id,
                "domain": chunk.get("domain") or documents_by_id[doc_id].get("domain"),
                "article": article,
            }
    return article_index


def _document_aliases(document: Dict[str, object]) -> List[str]:
    title = str(document.get("doc_title") or "")
    number = str(document.get("doc_number") or "")
    aliases = {title, number}
    lowered = _strip_accents(title)
    for token in ["luat ", "nghi dinh ", "thong tu ", "bo luat "]:
        if lowered.startswith(token):
            aliases.add(lowered[len(token) :])
    return [alias for alias in aliases if alias]


def _infer_domain_scores(text: str, taxonomy: Dict[str, dict]) -> List[Tuple[str, float]]:
    normalized = _strip_accents(text)
    scores: List[Tuple[str, float]] = []
    for domain, meta in taxonomy.items():
        keywords = [_strip_accents(keyword) for keyword in meta.get("keywords", [])]
        hits = sum(1 for keyword in keywords if keyword and keyword in normalized)
        if hits:
            scores.append((domain, float(hits) / float(max(len(keywords), 1))))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def enrich_references(
    chunks: List[Dict[str, object]],
    documents: List[Dict[str, object]],
    taxonomy: Dict[str, dict],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    documents_by_id = {str(document["doc_id"]): document for document in documents}
    article_index = _build_article_index(chunks, documents_by_id)
    alias_to_doc_id: Dict[str, str] = {}
    for document in documents:
        for alias in _document_aliases(document):
            alias_to_doc_id[_strip_accents(alias)] = str(document["doc_id"])

    enriched_chunks: List[Dict[str, object]] = []
    explicit_refs: List[Dict[str, object]] = []
    cross_domain_edges: List[Dict[str, object]] = []

    for chunk in chunks:
        chunk = dict(chunk)
        document = documents_by_id[str(chunk["doc_id"])]
        chunk["doc_ref"] = _doc_ref(document)
        chunk["article_ref"] = _article_ref(document, chunk.get("article"))
        chunk_refs: List[Dict[str, object]] = []
        content = str(chunk.get("content") or "")
        normalized_content = _strip_accents(content)

        referenced_doc_id = str(chunk["doc_id"])
        for alias, doc_id in alias_to_doc_id.items():
            if alias and alias in normalized_content and doc_id != str(chunk["doc_id"]):
                referenced_doc_id = doc_id
                break

        seen_keys = set()
        for match in ARTICLE_REF_RE.finditer(content):
            article = _normalize_article_label(match.group(0))
            if referenced_doc_id == str(chunk["doc_id"]) and article == chunk.get("article"):
                continue
            key = (referenced_doc_id, article)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            target = article_index.get(key)
            target_document = documents_by_id.get(referenced_doc_id, document)
            ref = {
                "source_chunk_id": chunk["chunk_id"],
                "source_doc_id": chunk["doc_id"],
                "source_domain": chunk["domain"],
                "match_text": match.group(0),
                "target_doc_id": referenced_doc_id,
                "target_doc_ref": _doc_ref(target_document),
                "target_article": article,
                "target_article_ref": _article_ref(target_document, article),
                "target_chunk_id": target["chunk_id"] if target else None,
                "target_context_chunk_id": target["context_chunk_id"] if target else None,
                "target_domain": target_document.get("domain"),
                "ref_type": "article",
                "resolution": "resolved" if target else "unresolved",
                "is_cross_doc": referenced_doc_id != str(chunk["doc_id"]),
                "is_cross_domain": (target_document.get("domain") or chunk["domain"]) != chunk["domain"],
                "confidence": 1.0 if target else 0.4,
            }
            chunk_refs.append(ref)
            explicit_refs.append(ref)
            if ref["is_cross_domain"]:
                cross_domain_edges.append(
                    {
                        "source_id": chunk["chunk_id"],
                        "target_id": ref["target_chunk_id"] or ref["target_doc_ref"],
                        "relation_type": "CITES_CROSS_DOMAIN",
                        "confidence": ref["confidence"],
                        "source_domain": chunk["domain"],
                        "target_domain": ref["target_domain"],
                        "match_text": ref["match_text"],
                    }
                )

        inferred_domains = _infer_domain_scores(f"{document.get('doc_title', '')}\n{content}", taxonomy)
        for domain, score in inferred_domains:
            if domain == chunk["domain"]:
                continue
            cross_domain_edges.append(
                {
                    "source_id": chunk["chunk_id"],
                    "target_id": f"domain:{domain}",
                    "relation_type": "RELATED_DOMAIN",
                    "confidence": round(score, 4),
                    "source_domain": chunk["domain"],
                    "target_domain": domain,
                    "match_text": None,
                }
            )
            break

        chunk["explicit_refs"] = chunk_refs
        enriched_chunks.append(chunk)

    return enriched_chunks, explicit_refs, cross_domain_edges


def run_reference_enricher(
    *,
    documents_path: str | Path = DEFAULT_DOCUMENTS_PATH,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    explicit_refs_path: str | Path = DEFAULT_EXPLICIT_REFS_PATH,
    cross_domain_edges_path: str | Path = DEFAULT_CROSS_DOMAIN_EDGES_PATH,
) -> Tuple[int, int]:
    documents = read_jsonl(documents_path)
    chunks = read_jsonl(chunks_path)
    taxonomy = _load_taxonomy(taxonomy_path)
    enriched_chunks, explicit_refs, cross_domain_edges = enrich_references(chunks, documents, taxonomy)
    write_jsonl(chunks_path, enriched_chunks)
    write_jsonl(explicit_refs_path, explicit_refs)
    write_jsonl(cross_domain_edges_path, cross_domain_edges)
    return len(explicit_refs), len(cross_domain_edges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich chunks with explicit legal references and cross-domain edges.")
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS_PATH))
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--explicit-refs", default=str(DEFAULT_EXPLICIT_REFS_PATH))
    parser.add_argument("--cross-domain-edges", default=str(DEFAULT_CROSS_DOMAIN_EDGES_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit_ref_count, cross_domain_count = run_reference_enricher(
        documents_path=args.documents,
        chunks_path=args.chunks,
        taxonomy_path=args.taxonomy,
        explicit_refs_path=args.explicit_refs,
        cross_domain_edges_path=args.cross_domain_edges,
    )
    print(f"Reference enrichment: DONE ({explicit_ref_count} explicit refs, {cross_domain_count} cross-domain edges)")


if __name__ == "__main__":
    main()
