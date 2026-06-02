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

from legal_rag.submission import validate_submission_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate results.json submission format.")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    report = validate_submission_file(args.input)
    print(f"Submission items: {report.item_count}")
    if report.errors:
        for error in report.errors:
            print(f"ERROR: {error}")
    if report.warnings:
        for warning in report.warnings:
            print(f"WARN: {warning}")
    if not report.ok:
        sys.exit(1)
    print("Submission validation passed.")


if __name__ == "__main__":
    main()
