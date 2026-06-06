import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion.bm25_builder import build_bm25_artifacts
from src.ingestion.incremental_update import diff_manifest
from src.ingestion.reference_enricher import enrich_references
from src.ingestion.sanity_report import build_sanity_report


class TestIngestionExtensions(unittest.TestCase):
    def test_reference_enricher_adds_doc_and_article_refs(self):
        documents = [
            {
                "doc_id": "doc-a",
                "doc_number": "01/2024/QH15",
                "doc_title": "Luật A",
                "domain": "business_law",
            },
            {
                "doc_id": "doc-b",
                "doc_number": "02/2024/QH15",
                "doc_title": "Luật B",
                "domain": "tax_law",
            },
        ]
        chunks = [
            {
                "chunk_id": "c1",
                "doc_id": "doc-a",
                "domain": "business_law",
                "article": "Điều 1",
                "level": "article",
                "content": "Theo Luật B và Điều 2 thì áp dụng khác.",
                "context_chunk_id": "ctx1",
            },
            {
                "chunk_id": "c2",
                "doc_id": "doc-b",
                "domain": "tax_law",
                "article": "Điều 2",
                "level": "article",
                "content": "Điều 2. Nội dung",
                "context_chunk_id": "ctx2",
            },
        ]
        taxonomy = {
            "business_law": {"keywords": ["doanh nghiệp"]},
            "tax_law": {"keywords": ["thuế", "luật b"]},
        }
        enriched_chunks, explicit_refs, cross_domain_edges = enrich_references(chunks, documents, taxonomy)
        self.assertTrue(enriched_chunks[0]["doc_ref"])
        self.assertTrue(enriched_chunks[0]["article_ref"])
        self.assertEqual(explicit_refs[0]["target_doc_id"], "doc-b")
        self.assertTrue(cross_domain_edges)

    def test_bm25_builder_generates_tokens(self):
        corpus, metadata = build_bm25_artifacts(
            [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc-a",
                    "domain": "business_law",
                    "citation": "Luật A, Điều 1",
                    "content": "thành lập doanh nghiệp",
                    "embedding_text": "Luật A, Điều 1\nthành lập doanh nghiệp",
                    "context_chunk_id": "ctx1",
                }
            ]
        )
        self.assertEqual(len(corpus), 1)
        self.assertTrue(corpus[0]["tokens"])
        self.assertEqual(metadata[0]["chunk_id"], "c1")

    def test_sanity_report_flags_noise(self):
        report = build_sanity_report(
            documents=[{"doc_id": "doc-a", "domain": "business_law"}],
            chunks=[
                {
                    "chunk_id": "c1",
                    "parent_id": "n1",
                    "context_chunk_id": "ctx1",
                    "content": "dang theo doi",
                    "prev_chunk_id": None,
                    "next_chunk_id": None,
                }
            ],
            context_chunks=[{"context_chunk_id": "ctx1"}],
            edges=[],
            explicit_refs=[],
            cross_domain_edges=[],
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["critical_issues"]["noise_hits"], ["c1"])

    def test_incremental_diff_detects_added_changed_removed(self):
        manifest_rows = [
            {"doc_id": "a", "success": True, "content_hash": "1", "html_hash": "x"},
            {"doc_id": "b", "success": True, "content_hash": "2", "html_hash": "y"},
        ]
        previous_state = {
            "manifest": {
                "b": {"content_hash": "old", "html_hash": "y", "success": True},
                "c": {"content_hash": "3", "html_hash": "z", "success": True},
            }
        }
        added, changed, removed, current_map = diff_manifest(manifest_rows, previous_state)
        self.assertEqual(added, ["a"])
        self.assertEqual(changed, ["b"])
        self.assertEqual(removed, ["c"])
        self.assertEqual(current_map["a"]["content_hash"], "1")


if __name__ == "__main__":
    unittest.main()
