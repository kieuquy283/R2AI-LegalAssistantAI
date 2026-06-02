from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from rag.utils.io import load_json


def _extract_row(file_path: Path) -> Dict[str, Any] | None:
    payload = load_json(file_path, {})
    if not isinstance(payload, dict):
        return None

    metrics = dict(payload.get("metrics", {}) or {})
    if not metrics:
        return None

    top_k = int(metrics.get("top_k", 5) or 5)
    hit_key = f"hit@{top_k}"
    recall_key = f"recall@{top_k}"

    return {
        "model_name": metrics.get("model_name", file_path.stem),
        "hit": float(metrics.get(hit_key, 0.0) or 0.0),
        "recall": float(metrics.get(recall_key, 0.0) or 0.0),
        "mrr": float(metrics.get("mrr", 0.0) or 0.0),
        "avg_latency_seconds": float(metrics.get("avg_latency_seconds", 0.0) or 0.0),
        "output_file": file_path.name,
    }


def compare_eval_runs(
    input_dir: str = "logs/eval_runs",
    pattern: str = "*legal_top5.json",
    output_csv: str = "logs/eval_runs/comparison_legal_top5.csv",
) -> List[Dict[str, Any]]:
    input_path = Path(input_dir)
    rows: List[Dict[str, Any]] = []

    for file_path in sorted(input_path.glob(pattern)):
        row = _extract_row(file_path)
        if row is not None:
            rows.append(row)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "hit@5",
                "recall@5",
                "mrr",
                "avg_latency_seconds",
                "output_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_name": row["model_name"],
                    "hit@5": f"{row['hit']:.6f}",
                    "recall@5": f"{row['recall']:.6f}",
                    "mrr": f"{row['mrr']:.6f}",
                    "avg_latency_seconds": f"{row['avg_latency_seconds']:.6f}",
                    "output_file": row["output_file"],
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare evaluation run JSON files and export a CSV summary.",
    )
    parser.add_argument(
        "--input-dir",
        default="logs/eval_runs",
    )
    parser.add_argument(
        "--pattern",
        default="*legal_top5.json",
    )
    parser.add_argument(
        "--output-csv",
        default="logs/eval_runs/comparison_legal_top5.csv",
    )
    args = parser.parse_args()

    rows = compare_eval_runs(
        input_dir=args.input_dir,
        pattern=args.pattern,
        output_csv=args.output_csv,
    )

    print("model_name | hit@5 | recall@5 | mrr | avg_latency_seconds | output_file")
    for row in rows:
        print(
            f"{row['model_name']} | "
            f"{row['hit']:.4f} | "
            f"{row['recall']:.4f} | "
            f"{row['mrr']:.4f} | "
            f"{row['avg_latency_seconds']:.4f} | "
            f"{row['output_file']}"
        )


if __name__ == "__main__":
    main()
