# Project Audit

## Active modules
- module/file: `rag/config`, `rag/utils`, `rag/ingestion`, `rag/retrieval`, `rag/modules`, `rag/pipelines`
- purpose: Core Python retrieval, chunking, reranking, and pipeline orchestration kept from the current stack.
- used by: `scripts/*.py`, `tests/*.py`, and the new `src/legal_rag` compatibility layer.

- module/file: `scripts/build_index.py`, `scripts/prepare_legal_dataset.py`, `scripts/evaluate_model_*.py`
- purpose: Existing indexing and legal retrieval evaluation entry points.
- used by: Local CLI workflows and smoke tests.

- module/file: `data/Legal_Dataset_V1.json`, `data/legal_corpus_chunks.json`, `data/multiturn_evaluation_legal.json`
- purpose: Current legal corpus and evaluation assets.
- used by: Dataset preparation, retrieval evaluation, and upcoming submission pipeline work.

## Legacy / unused candidates
- module/file: `app/`
- reason: FastAPI chat API for interactive demo; not required for batch legal submission.
- action: move_to_legacy

- module/file: `chatRAG/`
- reason: Frontend chat UI and bundled node artifacts; outside the batch competition scope.
- action: move_to_legacy

- module/file: `scripts/chat_cli.py`, `scripts/chat_cli_modular.py`
- reason: Interactive chat tooling; useful for debugging but not part of submission flow.
- action: keep

- module/file: `scripts/run_legal_eval_all*.ps1`
- reason: Research automation for ablation runs; not part of the core submission workflow but still reproducible.
- action: keep

- module/file: `runtime_data/`, `runtime_indexes/`, root `__pycache__/`, `.pytest_cache/`, `_write_root_test.txt`
- reason: Generated artifacts or temporary workspace files.
- action: remove or ignore after verification

- module/file: `docs/project_cleanup_report.md`, `docs/codex_execution_plan_model_3_to_8.md`
- reason: Historical refactor notes from earlier iterations.
- action: keep

## Risky dependencies
- dependency: `langchain`, `langchain-community`, `langchain-openai`, `langchain-core`, `langchain-text-splitters`
- reason: Heavy framework dependencies that are no longer central to the current code path and conflict with the requirement to avoid framework migration.
- replacement plan: Keep temporarily to preserve environment compatibility; assess safe removal only after Task 2 tests prove they are unused.

- dependency: `openai`
- reason: Current generation/rewrite helpers may depend on remote API configuration, which is not guaranteed to be competition-safe.
- replacement plan: Retain dependency for compatibility but route new legal answer-generation logic through controlled adapters and documented configuration.

- dependency: `torch`, `sentence-transformers`, `faiss-cpu`, `rank-bm25`
- reason: Core local retrieval stack; large install surface and model/runtime sensitivity.
- replacement plan: Keep unchanged and build new legal layers on top of the same stack.
