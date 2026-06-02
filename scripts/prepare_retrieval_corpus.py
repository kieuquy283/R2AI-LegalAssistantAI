from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from langchain_core.documents import Document

from rag.utils.io import save_json


def build_retrieval_corpus(
    dataset_name: str = "YuITC/Vietnamese-Legal-Documents",
    split: str = "train",
    output_json: str = "data/retrieval_corpus.json",
):
    ds = load_dataset(dataset_name, split=split)

    corpus = []
    documents = []
    seen_texts = set()

    for row in ds:
        context_list = row.get("context_list", [])
        cid_list = row.get("cid", [])

        if not isinstance(context_list, list):
            continue

        if not isinstance(cid_list, list):
            cid_list = [None] * len(context_list)

        for i, ctx in enumerate(context_list):
            if not isinstance(ctx, str):
                continue

            text = ctx.strip()
            if not text or text in seen_texts:
                continue

            seen_texts.add(text)
            cid = cid_list[i] if i < len(cid_list) else None

            item = {
                "text": text,
                "metadata": {
                    "cid": cid,
                    "source": dataset_name,
                    "split": split,
                },
            }
            corpus.append(item)
            documents.append(Document(page_content=text, metadata=item["metadata"]))

    save_json(output_json, corpus)
    print(f"Saved corpus to: {output_json}")
    print(f"Total unique contexts: {len(corpus)}")

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare retrieval corpus from HF dataset.")
    parser.add_argument("--dataset-name", default="YuITC/Vietnamese-Legal-Documents")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-json", default="data/retrieval_corpus.json")
    args = parser.parse_args()

    build_retrieval_corpus(
        dataset_name=args.dataset_name,
        split=args.split,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
