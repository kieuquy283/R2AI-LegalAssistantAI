from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion.merge_legal_corpora import merge_legal_corpora


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


class MergeLegalCorporaTests(unittest.TestCase):
    def test_merge_keeps_base_and_adds_hf_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _write_jsonl(base / "documents.jsonl", [{"doc_id": "base-doc"}])
            _write_jsonl(base / "nodes.jsonl", [{"node_id": "base-node"}])
            _write_jsonl(base / "chunks.jsonl", [{"chunk_id": "base-chunk"}])
            _write_jsonl(base / "context.jsonl", [{"context_chunk_id": "base-context"}])
            _write_jsonl(base / "edges.jsonl", [{"source_id": "a", "target_id": "b", "relation_type": "NEXT"}])

            _write_jsonl(base / "hf_documents.jsonl", [{"doc_id": "hf-doc"}])
            _write_jsonl(base / "hf_nodes.jsonl", [{"node_id": "hf-node"}])
            _write_jsonl(base / "hf_chunks.jsonl", [{"chunk_id": "hf-chunk"}])
            _write_jsonl(base / "hf_context.jsonl", [{"context_chunk_id": "hf-context"}])
            _write_jsonl(base / "hf_edges.jsonl", [{"source_id": "c", "target_id": "d", "relation_type": "NEXT"}])

            report = merge_legal_corpora(
                base_documents=base / "documents.jsonl",
                base_nodes=base / "nodes.jsonl",
                base_chunks=base / "chunks.jsonl",
                base_context_chunks=base / "context.jsonl",
                base_edges=base / "edges.jsonl",
                hf_documents=base / "hf_documents.jsonl",
                hf_nodes=base / "hf_nodes.jsonl",
                hf_chunks=base / "hf_chunks.jsonl",
                hf_context_chunks=base / "hf_context.jsonl",
                hf_edges=base / "hf_edges.jsonl",
                output_prefix=base / "merged",
            )

            self.assertEqual(report["merged_documents"], 2)
            self.assertEqual(report["merged_nodes"], 2)
            self.assertEqual(report["merged_chunks"], 2)


if __name__ == "__main__":
    unittest.main()
