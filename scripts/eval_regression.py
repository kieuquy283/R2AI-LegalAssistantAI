"""Task 10: Automated regression testing.

Usage:
    python scripts/eval_regression.py --questions-file data/evaluation/r2ai_stage1_questions.jsonl --output regression_report.json
    python scripts/eval_regression.py --sample 50 --output regression_report.json --verbose
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

try:
    from src.qa_pipeline import LegalQAPipeline
except ImportError:
    print("Cannot import LegalQAPipeline. Make sure you're running from project root.")
    sys.exit(1)


def load_questions(path: str, sample: int | None = None) -> list[dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            questions.append(entry)
    if sample and sample < len(questions):
        import random
        random.seed(42)
        questions = random.sample(questions, sample)
    return questions


def run_eval(questions: list[dict], verbose: bool = False) -> dict:
    pipeline = LegalQAPipeline()
    results = {
        "total": len(questions),
        "success": 0,
        "failed": 0,
        "empty_contexts": 0,
        "crag_used": 0,
        "low_confidence": 0,
        "avg_contexts": 0.0,
        "avg_docs": 0.0,
        "avg_time": 0.0,
        "details": [],
    }
    for i, entry in enumerate(questions):
        q = entry.get("question", "")
        if not q:
            continue
        t0 = time.perf_counter()
        try:
            result = pipeline.answer(q)
            elapsed = time.perf_counter() - t0
            results["success"] += 1

            detail = {
                "id": entry.get("id", i),
                "question": q[:120],
                "route": result.get("route"),
                "n_contexts": len(result.get("final_contexts", [])),
                "n_docs": len(result.get("relevant_docs", [])),
                "n_articles": len(result.get("relevant_articles", [])),
                "low_confidence": result.get("low_confidence", True),
                "crag_used": result.get("crag_used", False),
                "elapsed_s": round(elapsed, 3),
            }

            if not result.get("final_contexts"):
                results["empty_contexts"] += 1
                detail["empty"] = True
            if result.get("low_confidence"):
                results["low_confidence"] += 1
            if result.get("crag_used"):
                results["crag_used"] += 1

            results["avg_contexts"] += detail["n_contexts"]
            results["avg_docs"] += detail["n_docs"]
            results["avg_time"] += elapsed
            results["details"].append(detail)

            if verbose:
                status = "OK" if detail["n_contexts"] > 0 else "EMPTY"
                print(f"  [{i+1}/{results['total']}] {status} route={detail['route']} ctx={detail['n_contexts']} docs={detail['n_docs']} t={elapsed:.2f}s")
                if detail["n_contexts"] == 0:
                    print(f"    Question: {q[:120]}")
        except Exception as exc:
            results["failed"] += 1
            results["details"].append({
                "id": entry.get("id", i),
                "question": q[:120],
                "error": str(exc)[:200],
            })
            print(f"  [{i+1}/{results['total']}] FAILED: {exc}")

    if results["success"] > 0:
        results["avg_contexts"] = round(results["avg_contexts"] / results["success"], 2)
        results["avg_docs"] = round(results["avg_docs"] / results["success"], 2)
        results["avg_time"] = round(results["avg_time"] / results["success"], 2)
        results["coverage"] = round((results["success"] - results["empty_contexts"]) / results["success"] * 100, 1)

    return results


def main():
    parser = argparse.ArgumentParser(description="Regression testing for R2AI legal QA pipeline")
    parser.add_argument("--questions-file", default="data/evaluation/r2ai_stage1_questions.jsonl",
                        help="Path to questions JSONL")
    parser.add_argument("--output", default="regression_report.json",
                        help="Output report path")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to sample (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question progress")
    args = parser.parse_args()

    questions = load_questions(args.questions_file, sample=args.sample)
    print(f"Loaded {len(questions)} questions from {args.questions_file}")

    results = run_eval(questions, verbose=args.verbose)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"REGRESSION REPORT — {args.output}")
    print(f"{'='*50}")
    print(f"  Total:      {results['total']}")
    print(f"  Success:    {results['success']}")
    print(f"  Failed:     {results['failed']}")
    print(f"  Coverage:   {results.get('coverage', 0)}% (non-empty)")
    print(f"  Low conf:   {results['low_confidence']}")
    print(f"  CRAG used:  {results['crag_used']}")
    print(f"  Avg ctxs:   {results['avg_contexts']}")
    print(f"  Avg docs:   {results['avg_docs']}")
    print(f"  Avg time:   {results['avg_time']}s")


if __name__ == "__main__":
    main()
