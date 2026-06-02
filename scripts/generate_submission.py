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

from legal_rag.aggregation import ArticleAggregator
from legal_rag.corpus import load_articles_jsonl
from legal_rag.generation import LegalAnswerGenerator, ensure_citations
from legal_rag.retrieval import KeywordArticleRetriever
from legal_rag.submission import SubmissionItem, export_submission
from legal_rag.utils import load_json


def parse_question(raw_item: dict, fallback_id: int) -> tuple[int, str]:
    question_id = int(raw_item.get("id", fallback_id))
    question = str(raw_item.get("question") or raw_item.get("current_question") or "").strip()
    if not question:
        raise ValueError(f"Question item {question_id} is missing question text.")
    return question_id, question


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate competition submission results.json.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--metadata", default="data/processed/articles.jsonl")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    questions = load_json(args.input, [])
    articles = load_articles_jsonl(args.metadata)
    retriever = KeywordArticleRetriever(articles)
    aggregator = ArticleAggregator()
    generator = LegalAnswerGenerator()

    submission_items: list[SubmissionItem] = []
    for index, raw_item in enumerate(questions, start=1):
        question_id, question = parse_question(raw_item, index)
        retrieved = retriever.retrieve(question, top_k=args.top_k)
        selected = aggregator.select(retrieved, query=question)
        answer = ensure_citations(generator.generate(question, selected), selected)
        submission_items.append(
            SubmissionItem(
                id=question_id,
                question=question,
                answer=answer,
                relevant_docs=list(dict.fromkeys(article.doc_ref for article in selected)),
                relevant_articles=list(dict.fromkeys(article.article_id for article in selected)),
            )
        )

    export_submission(submission_items, args.output)
    print(f"Generated {len(submission_items)} submission items at {Path(args.output)}")


if __name__ == "__main__":
    main()
