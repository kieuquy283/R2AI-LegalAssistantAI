from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


class ErrorAnalyzer:
    def analyze_records(self, records: List[Dict[str, object]], *, run_id: str = "ad_hoc") -> Dict[str, object]:
        route_distribution: Counter[str] = Counter()
        error_counts: Counter[str] = Counter()

        missing_citation = 0
        no_context = 0
        empty_answer = 0
        low_grounding = 0
        cross_domain_missing = 0
        possible_hallucination = 0

        for record in records:
            route = str(record.get("route") or "")
            route_distribution[route] += 1

            citations = list(record.get("citations") or [])
            final_context_ids = list(record.get("final_context_ids") or [])
            answer = str(record.get("answer") or "").strip()
            grounding = dict(record.get("grounding") or {})
            domains = list(record.get("domains") or [])

            if not citations:
                missing_citation += 1
                error_counts["missing_citation"] += 1
            if not final_context_ids:
                no_context += 1
                error_counts["no_context"] += 1
            if not answer:
                empty_answer += 1
                error_counts["empty_answer"] += 1
            if grounding and not grounding.get("is_grounded", True):
                low_grounding += 1
                error_counts["low_grounding"] += 1
            if grounding and grounding.get("unsupported_claims"):
                possible_hallucination += 1
                error_counts["possible_hallucination"] += 1
            if route == "CROSS_DOMAIN_CONTEXT" and len(domains) <= 1:
                cross_domain_missing += 1
                error_counts["cross_domain_missing"] += 1

        return {
            "run_id": run_id,
            "total_records": len(records),
            "missing_citation": missing_citation,
            "no_context": no_context,
            "empty_answer": empty_answer,
            "low_grounding": low_grounding,
            "cross_domain_missing": cross_domain_missing,
            "possible_hallucination": possible_hallucination,
            "route_distribution": dict(route_distribution),
            "top_errors": [{"error": name, "count": count} for name, count in error_counts.most_common(10)],
        }

    def analyze_file(self, input_path: str | Path) -> Dict[str, object]:
        path = Path(input_path)
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        report = self.analyze_records(records, run_id=path.stem)
        output_path = path.with_name(f"{path.stem}_error_analysis.json")
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def analyze_run(self, run_id: str, *, logs_dir: str | Path = "logs/eval_runs") -> Dict[str, object]:
        path = Path(logs_dir) / f"{run_id}.jsonl"
        return self.analyze_file(path)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Analyze eval logs for citation, context, and grounding issues.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    analyzer = ErrorAnalyzer()
    if args.input:
        report = analyzer.analyze_file(args.input)
    elif args.run_id:
        report = analyzer.analyze_run(args.run_id)
    else:
        raise SystemExit("Provide --run-id or --input")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
