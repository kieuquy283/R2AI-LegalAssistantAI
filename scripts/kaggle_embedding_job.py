"""
Kaggle Embedding Job
====================
Standalone script to embed Vietnamese legal chunks on Kaggle GPU (T4).

Usage on Kaggle:
    !pip install sentence-transformers
    !python kaggle_embedding_job.py \
        --input /kaggle/input/legal-chunks/legal_chunks_to_embed.jsonl \
        --output /kaggle/working/legal_chunks_embedded.jsonl \
        --model BAAI/bge-m3 \
        --batch-size 32

Input format (JSON Lines):
    {"id": "...", "text": "...", "metadata": {...}}

Output format (JSON Lines):
    {"id": "...", "vector": [0.01, ...], "metadata": {...}}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[Dict[str, Any]], append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def embed_chunks(
    input_path: str,
    output_path: str,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 32,
    device: str = "cuda",
    max_seq_length: int = 512,
) -> None:
    print(f"[INFO] Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length

    print(f"[INFO] Reading input: {input_path}")
    records = read_jsonl(input_path)
    print(f"[INFO] Total records to embed: {len(records)}")

    texts = [rec["text"] for rec in records]
    ids = [rec["id"] for rec in records]
    metadatas = [rec.get("metadata", {}) for rec in records]

    print(f"[INFO] Starting embedding (batch_size={batch_size}, device={device})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )

    print(f"[INFO] Writing output: {output_path}")
    output_records = []
    for idx, vec in enumerate(embeddings):
        output_records.append({
            "id": ids[idx],
            "vector": vec.tolist(),
            "metadata": metadatas[idx],
        })
    write_jsonl(output_path, output_records)
    print("[INFO] Embedding job completed.")


def embed_chunks_streaming(
    input_path: str,
    output_path: str,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 32,
    device: str = "cuda",
    max_seq_length: int = 512,
) -> None:
    """Streaming version with checkpointing (append mode) for very large files."""
    print(f"[INFO] Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length

    # Count total lines for progress bar
    total_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    processed_ids = set()
    if os.path.exists(output_path):
        print(f"[INFO] Found existing output. Loading processed IDs...")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed_ids.add(rec["id"])
                except Exception:
                    continue
        print(f"[INFO] Resuming from {len(processed_ids)} already embedded records.")

    print(f"[INFO] Embedding (batch_size={batch_size}, device={device})...")
    batch_records: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f_in, tqdm(total=total_lines, desc="Embedding") as pbar:
        for line in f_in:
            pbar.update(1)
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["id"] in processed_ids:
                continue
            batch_records.append(rec)
            if len(batch_records) >= batch_size:
                _process_batch(batch_records, model, output_path)
                batch_records = []
        if batch_records:
            _process_batch(batch_records, model, output_path)

    print("[INFO] Streaming embedding job completed.")


def _process_batch(
    records: List[Dict[str, Any]],
    model: SentenceTransformer,
    output_path: str,
) -> None:
    texts = [rec["text"] for rec in records]
    embeddings = model.encode(
        texts,
        batch_size=len(texts),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    out = []
    for i, vec in enumerate(embeddings):
        out.append({
            "id": records[i]["id"],
            "vector": vec.tolist(),
            "metadata": records[i].get("metadata", {}),
        })
    write_jsonl(output_path, out, append=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed legal chunks on Kaggle GPU.")
    parser.add_argument("--input", required=True, help="Path to legal_chunks_to_embed.jsonl")
    parser.add_argument("--output", required=True, help="Path to legal_chunks_embedded.jsonl")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode with checkpointing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.streaming:
        embed_chunks_streaming(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            max_seq_length=args.max_seq_length,
        )
    else:
        embed_chunks(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            max_seq_length=args.max_seq_length,
        )


if __name__ == "__main__":
    main()
