# Architecture Overview

## Goal

Refactor the repository into a legal-assistant codebase that preserves the current Python RAG stack while separating active submission code from demo or historical components.

## Active runtime

The active backend remains the existing `rag/` package:

- `rag/ingestion`: document loading, chunking, metadata, index build helpers
- `rag/modules/retrieval`: dense, sparse, and hybrid retrieval modules
- `rag/modules/reranking`: reranker abstractions
- `rag/modules/query_rewriting`: rewrite and multi-query logic
- `rag/pipelines`: legacy, modular, and adaptive orchestration

## Teamwork layer

The new `src/legal_rag/` package is the stable surface for legal-assistant development:

- `legal_rag.config`
- `legal_rag.corpus`
- `legal_rag.retrieval`
- `legal_rag.reranking`
- `legal_rag.aggregation`
- `legal_rag.generation`
- `legal_rag.evaluation`
- `legal_rag.submission`
- `legal_rag.utils`

During Task 1 this layer primarily provides package structure and compatibility exports. Task 2 will move legal-specific schemas, aggregation, answer generation, evaluation, and submission logic into this package.

## Legacy isolation

Interactive demo surfaces are separated from the core pipeline:

- `legacy/app/`: legacy FastAPI chat API
- `legacy/chatRAG/`: legacy frontend workspace

This keeps the legal batch pipeline focused while preserving old assets for reference.

## Data layout

- `data/raw/`: raw datasets and sample questions
- `data/processed/`: normalized chunks, metadata, and evaluation gold
- `data/indexes/`: index artifacts
- `data/submissions/`: generated `results.json` and evaluation reports

## Compatibility approach

- Keep `rag/` imports working for the existing scripts and tests.
- Add `src/legal_rag/` wrappers so new code uses package-based imports without forcing a framework migration.
- Update documentation and config defaults toward the legal competition workflow.
