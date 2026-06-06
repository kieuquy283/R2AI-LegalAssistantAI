from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List


class QueryAnalytics:
    def _tokenize(self, text: str) -> List[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()) if len(token) >= 3]

    def analyze_eval_logs(self, logs_dir: str | Path = "logs/eval_runs") -> Dict[str, object]:
        route_distribution: Counter[str] = Counter()
        domain_distribution: Counter[str] = Counter()
        term_distribution: Counter[str] = Counter()
        total_queries = 0
        total_contexts = 0
        missing_context_count = 0
        missing_citation_count = 0

        for path in sorted(Path(logs_dir).glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = json.loads(stripped)
                    total_queries += 1
                    route_distribution[str(record.get("route") or "UNKNOWN")] += 1
                    for domain in record.get("domains") or []:
                        domain_distribution[str(domain)] += 1
                    final_context_ids = list(record.get("final_context_ids") or [])
                    total_contexts += len(final_context_ids)
                    if not final_context_ids:
                        missing_context_count += 1
                    if not list(record.get("citations") or []):
                        missing_citation_count += 1
                    for token in self._tokenize(str(record.get("question") or "")):
                        term_distribution[token] += 1

        report = {
            "total_queries": total_queries,
            "route_distribution": dict(route_distribution),
            "domain_distribution": dict(domain_distribution),
            "avg_context_count": (total_contexts / total_queries) if total_queries else 0.0,
            "missing_context_count": missing_context_count,
            "missing_citation_count": missing_citation_count,
            "top_terms": [{"term": term, "count": count} for term, count in term_distribution.most_common(10)],
        }
        output_path = Path("logs/analytics/query_analytics.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Aggregate query analytics from eval logs.")
    parser.add_argument("--logs-dir", default="logs/eval_runs")
    args = parser.parse_args()
    report = QueryAnalytics().analyze_eval_logs(args.logs_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
