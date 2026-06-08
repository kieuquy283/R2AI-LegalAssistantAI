from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if completed.returncode != 0:  # pragma: no cover - subprocess.run(check=True) raises first
        raise SystemExit(completed.returncode)


def _write_derived_gold(pred_path: Path, gold_path: Path) -> None:
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    gold_payload = [
        {
            "id": item["id"],
            "relevant_articles": item.get("relevant_articles", []),
        }
        for item in predictions
        if isinstance(item, dict) and "id" in item
    ]
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(json.dumps(gold_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full R2AI Stage 1 pipeline end-to-end.")
    parser.add_argument("--input", default="data/evaluation/R2AIStage1DATA.json")
    parser.add_argument("--questions", default="data/evaluation/r2ai_stage1_questions.jsonl")
    parser.add_argument("--submission", default="data/submissions/results.json")
    parser.add_argument("--gold", default="data/processed/r2ai_stage1_gold.json")
    parser.add_argument(
        "--derive-gold",
        action="store_true",
        help="Derive a smoke-test gold file from the generated submission if no gold file exists.",
    )
    parser.add_argument(
        "--report",
        default="data/submissions/r2ai_stage1_eval_report.json",
        help="Evaluation report output path.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Stop after validating the generated submission.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    questions_path = Path(args.questions)
    submission_path = Path(args.submission)
    gold_path = Path(args.gold)
    report_path = Path(args.report)

    _run(
        [
            sys.executable,
            "-m",
            "src.evaluation.prepare_r2ai_dataset",
            "--input",
            str(input_path),
            "--output",
            str(questions_path),
        ]
    )

    _run(
        [
            sys.executable,
            "scripts/generate_submission.py",
            "--input",
            str(questions_path),
            "--output",
            str(submission_path),
        ]
    )

    _run([sys.executable, "scripts/validate_submission.py", "--input", str(submission_path)])

    if args.validate_only:
        return

    if not gold_path.exists():
        if not args.derive_gold:
            print(f"Skipping evaluation because gold file is missing: {gold_path}")
            return
        _write_derived_gold(submission_path, gold_path)

    _run(
        [
            sys.executable,
            "scripts/evaluate_submission.py",
            "--pred",
            str(submission_path),
            "--gold",
            str(gold_path),
            "--output",
            str(report_path),
        ]
    )


if __name__ == "__main__":
    main()
