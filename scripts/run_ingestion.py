from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], check=True)


def _exists(path: str) -> bool:
    return Path(path).exists()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run the canonical ingestion pipeline with safe defaults.")
    parser.add_argument("--smoke", action="store_true", help="Only validate/build local artifacts and skip live crawl steps.")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip source registry, URL collection, and crawl steps.")
    args = parser.parse_args()

    skip_crawl = args.skip_crawl or args.smoke

    if not skip_crawl:
        print("Source and crawl steps are available but should only run when live crawling is explicitly intended.")
        _run(["-m", "src.ingestion.source_registry"])
        _run(["-m", "src.ingestion.collect_urls", "--limit", "5"])
        _run(["-m", "src.ingestion.crawl_documents", "--limit", "5"])

    required = ["data/raw/documents_manifest.jsonl"]
    missing = [path for path in required if not _exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing required ingestion inputs: {missing}")

    commands = [
        ["-m", "src.ingestion.document_parser", "--manifest", "data/raw/documents_manifest.jsonl", "--output", "data/processed/documents.jsonl"],
        ["-m", "src.ingestion.text_cleaner", "--input", "data/processed/documents.jsonl", "--output", "data/processed/cleaned_documents.jsonl"],
        ["-m", "src.ingestion.legal_structure_parser", "--input", "data/processed/cleaned_documents.jsonl", "--output", "data/processed/legal_nodes.jsonl"],
        ["-m", "src.ingestion.legal_chunker", "--nodes", "data/processed/legal_nodes.jsonl", "--documents", "data/processed/documents.jsonl", "--output", "data/processed/chunks.jsonl", "--context-output", "data/processed/context_chunks.jsonl", "--edges-output", "data/processed/legal_edges.jsonl"],
        ["-m", "src.ingestion.reference_enricher", "--documents", "data/processed/documents.jsonl", "--chunks", "data/processed/chunks.jsonl", "--taxonomy", "data/sources/domain_taxonomy.json", "--explicit-refs", "data/processed/explicit_refs.jsonl", "--cross-domain-edges", "data/processed/cross_domain_edges.jsonl"],
        ["-m", "src.ingestion.graph_builder", "--documents", "data/processed/documents.jsonl", "--chunks", "data/processed/chunks.jsonl", "--nodes", "data/processed/legal_nodes.jsonl", "--edges", "data/processed/legal_edges.jsonl", "--explicit-refs", "data/processed/explicit_refs.jsonl", "--cross-domain-edges", "data/processed/cross_domain_edges.jsonl", "--domain-taxonomy", "data/sources/domain_taxonomy.json", "--output-nodes", "data/processed/legal_graph_nodes.jsonl", "--output-edges", "data/processed/legal_graph_edges.jsonl"],
        ["-m", "src.ingestion.bm25_builder"],
        ["-m", "src.ingestion.index_builder"],
        ["-m", "src.ingestion.sanity_report"],
    ]

    for command in commands:
        _run(command)

    print("Canonical ingestion run completed.")


if __name__ == "__main__":
    main()
