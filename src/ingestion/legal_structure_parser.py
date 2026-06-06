from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.ingestion.common import read_jsonl, stable_slug, write_jsonl


DEFAULT_INPUT_PATH = Path("data/processed/cleaned_documents.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/legal_nodes.jsonl")

CHAPTER_RE = re.compile(r"^\**\s*Chương\s+([IVXLCDM0-9]+)\b.*$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\**\s*Mục\s+(\d+)\b.*$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^\**\s*Điều\s+(\d+[A-Za-z]?)\.\s*(.*)$", re.IGNORECASE)
CLAUSE_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")
POINT_RE = re.compile(r"^\s*([a-zđ])\)\s+(.*)$", re.IGNORECASE)


def _make_node_id(doc_id: str, level: str, *parts: str) -> str:
    key = "_".join(part for part in parts if part)
    return stable_slug(f"{doc_id}_{level}_{key}")


def _finalize_node(
    nodes: List[Dict[str, object]],
    node: Optional[Dict[str, object]],
    end_char: int,
) -> None:
    if not node:
        return
    node["content"] = str(node.get("content", "")).strip()
    node["end_char"] = max(end_char, int(node.get("start_char", 0)))
    if node["content"] or node.get("title"):
        nodes.append(node)


def parse_document_structure(document: Dict[str, str]) -> List[Dict[str, object]]:
    text = document["cleaned_text"]
    nodes: List[Dict[str, object]] = []
    current_chapter: Optional[Dict[str, object]] = None
    current_section: Optional[Dict[str, object]] = None
    current_article: Optional[Dict[str, object]] = None
    current_clause: Optional[Dict[str, object]] = None
    current_point: Optional[Dict[str, object]] = None
    cursor = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cursor += len(raw_line) + 1
            continue

        chapter_match = CHAPTER_RE.match(line)
        section_match = SECTION_RE.match(line)
        article_match = ARTICLE_RE.match(line)
        clause_match = CLAUSE_RE.match(line)
        point_match = POINT_RE.match(line)

        if chapter_match:
            _finalize_node(nodes, current_point, cursor)
            _finalize_node(nodes, current_clause, cursor)
            _finalize_node(nodes, current_article, cursor)
            _finalize_node(nodes, current_section, cursor)
            _finalize_node(nodes, current_chapter, cursor)
            numeral = chapter_match.group(1).upper()
            current_chapter = {
                "node_id": _make_node_id(document["doc_id"], "chapter", numeral),
                "doc_id": document["doc_id"],
                "level": "chapter",
                "title": line,
                "article": None,
                "article_title": None,
                "clause": None,
                "point": None,
                "content": "",
                "parent_id": None,
                "start_char": cursor,
                "end_char": cursor,
                "domain": document["domain"],
                "source_url": document["source_url"],
            }
            current_section = None
            current_article = None
            current_clause = None
            current_point = None
        elif section_match:
            _finalize_node(nodes, current_point, cursor)
            _finalize_node(nodes, current_clause, cursor)
            _finalize_node(nodes, current_article, cursor)
            _finalize_node(nodes, current_section, cursor)
            section_number = section_match.group(1)
            current_section = {
                "node_id": _make_node_id(document["doc_id"], "section", section_number),
                "doc_id": document["doc_id"],
                "level": "section",
                "title": line,
                "article": None,
                "article_title": None,
                "clause": None,
                "point": None,
                "content": "",
                "parent_id": current_chapter["node_id"] if current_chapter else None,
                "start_char": cursor,
                "end_char": cursor,
                "domain": document["domain"],
                "source_url": document["source_url"],
            }
            current_article = None
            current_clause = None
            current_point = None
        elif article_match:
            _finalize_node(nodes, current_point, cursor)
            _finalize_node(nodes, current_clause, cursor)
            _finalize_node(nodes, current_article, cursor)
            article_number = article_match.group(1)
            article_title = article_match.group(2).strip() or None
            current_article = {
                "node_id": _make_node_id(document["doc_id"], "article", article_number),
                "doc_id": document["doc_id"],
                "level": "article",
                "title": line,
                "article": f"Điều {article_number}",
                "article_title": article_title,
                "clause": None,
                "point": None,
                "content": "",
                "parent_id": (current_section or current_chapter or {}).get("node_id"),
                "start_char": cursor,
                "end_char": cursor,
                "domain": document["domain"],
                "source_url": document["source_url"],
            }
            current_clause = None
            current_point = None
        elif clause_match and current_article:
            _finalize_node(nodes, current_point, cursor)
            _finalize_node(nodes, current_clause, cursor)
            clause_number = clause_match.group(1)
            current_clause = {
                "node_id": _make_node_id(document["doc_id"], "clause", current_article["article"], clause_number),
                "doc_id": document["doc_id"],
                "level": "clause",
                "title": f"Khoản {clause_number}",
                "article": current_article["article"],
                "article_title": current_article.get("article_title"),
                "clause": f"Khoản {clause_number}",
                "point": None,
                "content": line,
                "parent_id": current_article["node_id"],
                "start_char": cursor,
                "end_char": cursor,
                "domain": document["domain"],
                "source_url": document["source_url"],
            }
            current_point = None
        elif point_match and current_clause:
            _finalize_node(nodes, current_point, cursor)
            point_label = point_match.group(1)
            current_point = {
                "node_id": _make_node_id(
                    document["doc_id"],
                    "point",
                    str(current_article["article"]),
                    str(current_clause["clause"]),
                    point_label,
                ),
                "doc_id": document["doc_id"],
                "level": "point",
                "title": f"Điểm {point_label}",
                "article": current_article["article"],
                "article_title": current_article.get("article_title"),
                "clause": current_clause["clause"],
                "point": f"Điểm {point_label}",
                "content": line,
                "parent_id": current_clause["node_id"],
                "start_char": cursor,
                "end_char": cursor,
                "domain": document["domain"],
                "source_url": document["source_url"],
            }
        else:
            active_node = current_point or current_clause or current_article or current_section or current_chapter
            if active_node:
                active_node["content"] = f"{active_node.get('content', '')}\n{line}".strip()

        cursor += len(raw_line) + 1

    _finalize_node(nodes, current_point, cursor)
    _finalize_node(nodes, current_clause, cursor)
    _finalize_node(nodes, current_article, cursor)
    _finalize_node(nodes, current_section, cursor)
    _finalize_node(nodes, current_chapter, cursor)
    return nodes


def iter_legal_nodes(cleaned_documents_path: str | Path) -> Iterable[Dict[str, object]]:
    for document in read_jsonl(cleaned_documents_path):
        for node in parse_document_structure(document):
            yield node


def run_legal_structure_parser(
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> int:
    return write_jsonl(output_path, iter_legal_nodes(input_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse legal hierarchy from cleaned documents.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = run_legal_structure_parser(input_path=args.input, output_path=args.output)
    print(f"Legal structure parsing: DONE ({count} nodes)")


if __name__ == "__main__":
    main()
