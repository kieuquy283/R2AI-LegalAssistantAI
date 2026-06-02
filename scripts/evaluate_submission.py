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

from legal_rag.evaluation import evaluate_submission_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate submission article retrieval quality.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = evaluate_submission_files(args.pred, args.gold, args.output)
    print(
        "macro_precision={macro_precision:.4f} macro_recall={macro_recall:.4f} macro_f2={macro_f2:.4f} num_questions={num_questions}".format(
            **report
        )
    )


if __name__ == "__main__":
    main()
