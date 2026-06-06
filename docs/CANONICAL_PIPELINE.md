# Canonical Pipeline

## Scope

Repo này có hai lớp code:

- `src/`: canonical pipeline cho ingestion, retrieval, QA, evaluation
- `rag/` và một số script legacy: compatibility layer, utility cũ, và fallback embedding/vectorstore

Khi phát triển tính năng mới cho Hybrid RAG + Lightweight GraphRAG, ưu tiên đi theo `src/`.

## 1. Data Ingestion Path

Thứ tự chuẩn:

1. `src.ingestion.source_registry`
2. `src.ingestion.collect_urls`
3. `src.ingestion.crawl_documents`
4. `src.ingestion.document_parser`
5. `src.ingestion.text_cleaner`
6. `src.ingestion.legal_structure_parser`
7. `src.ingestion.legal_chunker`
8. `src.ingestion.reference_enricher`
9. `src.ingestion.graph_builder`
10. `src.ingestion.bm25_builder`
11. `src.ingestion.index_builder`
12. `src.ingestion.sanity_report`

Artifacts chuẩn:

- `data/raw/documents_manifest.jsonl`
- `data/processed/documents.jsonl`
- `data/processed/cleaned_documents.jsonl`
- `data/processed/legal_nodes.jsonl`
- `data/processed/chunks.jsonl`
- `data/processed/context_chunks.jsonl`
- `data/processed/legal_graph_nodes.jsonl`
- `data/processed/legal_graph_edges.jsonl`
- `data/indexes/faiss.index`
- `data/indexes/chunk_metadata.json`
- `data/indexes/bm25_corpus.json`

## 2. Graph / Index Build Path

Canonical graph/index build:

1. `reference_enricher` tạo `explicit_refs.jsonl` và `cross_domain_edges.jsonl`
2. `graph_builder` chuẩn hóa sang `legal_graph_nodes.jsonl` và `legal_graph_edges.jsonl`
3. `bm25_builder` build sparse corpus
4. `index_builder` build FAISS và chunk metadata

## 3. Retrieval Path

Canonical runtime:

1. `src.retrieval.hybrid_retriever.HybridRetriever`
2. `src.retrieval.query_router.route_query`
3. `src.retrieval.confidence_checker.ConfidenceChecker`
4. `src.retrieval.context_expander.ContextExpander`
5. `src.retrieval.reranker.Reranker`
6. `src.retrieval.retrieval_pipeline.RetrievalPipeline`

`rag/retrieval/vectorstore.py` hiện vẫn là compatibility layer cho embedding runtime và offline fallback.

## 4. QA Path

Canonical QA path:

1. `src.qa_pipeline.LegalQAPipeline`
2. `src.retrieval.retrieval_pipeline.RetrievalPipeline`
3. `src.generation.answer_generator.AnswerGenerator`

## 5. Evaluation Path

Canonical evaluation path:

1. `src.evaluation.eval_logger.EvalLogger`
2. `src.evaluation.evaluate_qa.evaluate_questions`
3. `logs/eval_runs/<run_id>.jsonl`
4. `logs/eval_runs/<run_id>_summary.json`

## 6. Legacy / Compatibility Notes

- `rag/`: giữ lại cho embedding/vectorstore compatibility và một số utility retrieval cũ.
- `src/legal_rag/`: code cũ giữ để tham chiếu, không phải bề mặt canonical cho task gap-fix hiện tại.
- `scripts/evaluate_model_*`: benchmark/eval legacy theo các pipeline đời trước.

## 7. Canonical Commands

Ingestion rebuild không crawl:

```bash
python scripts/run_ingestion.py --skip-crawl
```

Retrieval smoke:

```bash
python scripts/run_retrieval_smoke.py
```

QA smoke:

```bash
python scripts/run_qa_smoke.py
```

Evaluation smoke:

```bash
python scripts/run_eval_smoke.py
```
