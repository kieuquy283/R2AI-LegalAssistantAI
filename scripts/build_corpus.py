from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from legal_rag.corpus import normalize_corpus_items, save_articles_jsonl
from legal_rag.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize legal corpus chunks into article-level JSONL metadata.")
    parser.add_argument("--input", default="data/legal_corpus_chunks.json")
    parser.add_argument("--output", default="data/processed/articles.jsonl")
    args = parser.parse_args()

    items = load_json(args.input, [])
    articles = normalize_corpus_items(items)
    save_articles_jsonl(args.output, articles)
    print(f"Normalized {len(articles)} legal articles to {args.output}")


if __name__ == "__main__":
    main()
