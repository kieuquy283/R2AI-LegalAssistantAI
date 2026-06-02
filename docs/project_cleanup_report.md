# Project Cleanup Report

## Current Architecture Summary

The repository currently has two parallel architectures:

- A **legacy linear runtime path** used by the app, CLI, and older evaluation scripts.
- A **new modular research path** centered around `rag/modules/*`.

The runtime app path is still:

`question -> legacy query rewriting -> legacy retrieval -> active-doc filtering -> answer generation`

The new modular path currently exists mostly for research, ablation, and test coverage:

- `rag/modules/history_selection/*`
- `rag/modules/query_rewriting/*`
- `rag/modules/retrieval/*`
- `rag/modules/reranking/*`

## Old Linear Components

These files still represent the older linear architecture and remain active:

- `rag/retrieval/query_rewriter.py`
- `rag/retrieval/retriever.py`
- `rag/retrieval/ranking.py`
- `rag/retrieval/vectorstore.py`
- `rag/pipelines/chat_pipeline.py`
- `rag/generation/answering.py`
- `rag/generation/prompt_builder.py`
- `scripts/evaluate_retrieval.py`
- `scripts/evaluate_multiturn_retrieval.py`

## New Modular Components

These packages define the intended research architecture:

- `rag/modules/history_selection/`
- `rag/modules/query_rewriting/`
- `rag/modules/retrieval/`
- `rag/modules/reranking/`

There is also a modular generation package:

- `rag/modules/generation/`

At the moment, it appears present but not yet wired into the main serving path.

## Files That Appear Active

### App / Serving

- `app/api.py`
- `scripts/chat_cli.py`
- `rag/pipelines/chat_pipeline.py`

### Retrieval / Runtime Dependencies

- `rag/retrieval/query_rewriter.py`
- `rag/retrieval/retriever.py`
- `rag/retrieval/ranking.py`
- `rag/retrieval/vectorstore.py`
- `rag/generation/answering.py`

### Evaluation / Ablation

- `scripts/evaluate_retrieval.py`
- `scripts/evaluate_multiturn_retrieval.py`
- `scripts/evaluate_model_1_baseline.py`
- `scripts/evaluate_model_2_rewrite_dense.py`

### Data Preparation / Indexing

- `scripts/build_index.py`
- `scripts/prepare_legal_dataset.py`
- `rag/pipelines/indexing_pipeline.py`
- `rag/ingestion/indexing.py`

## Files That Appear Legacy

These files still matter, but they should now be treated as compatibility layers or older evaluators:

- `rag/retrieval/query_rewriter.py`
- `rag/pipelines/chat_pipeline.py`
- `scripts/evaluate_retrieval.py`
- `scripts/evaluate_multiturn_retrieval.py`

## Files That Should Be Kept

Keep these for backward compatibility:

- `rag/retrieval/query_rewriter.py`
- `rag/retrieval/retriever.py`
- `rag/retrieval/ranking.py`
- `rag/retrieval/vectorstore.py`
- `rag/pipelines/chat_pipeline.py`
- old evaluation scripts under `scripts/`

Reason:

- they are still imported by app/API/CLI/tests
- they preserve the current public behavior
- they provide the LangChain `Document`-based retrieval flow used by serving code

## Files That Can Be Deprecated

These should remain, but be clearly labeled as legacy or compatibility code:

- `rag/retrieval/query_rewriter.py`
- `rag/pipelines/chat_pipeline.py`
- `scripts/evaluate_retrieval.py`
- `scripts/evaluate_multiturn_retrieval.py`

## Files That Can Potentially Be Removed Later

Only after a second cleanup pass and import verification:

- duplicated helper logic embedded in old evaluation scripts
- standalone heuristic/query-formatting code duplicated between legacy and modular rewriting paths
- any unwired experimental package that remains unreferenced after migration

No source file is recommended for immediate deletion in the first cleanup pass.

## Main Duplication Areas

### Query Rewriting

Duplicated between:

- `rag/retrieval/query_rewriter.py`
- `rag/modules/query_rewriting/*`

### Retrieval Metrics

Duplicated between:

- `scripts/evaluate_retrieval.py`
- `scripts/evaluate_multiturn_retrieval.py`
- `scripts/evaluate_model_1_baseline.py`
- `scripts/evaluate_model_2_rewrite_dense.py`
- `app/api.py`

### History Formatting / Follow-up Heuristics

Duplicated between:

- legacy query rewriter
- modular query rewriting formatter/utils

## Risks and Compatibility Notes

- The legacy retrieval path returns LangChain `Document` objects.
- The modular retrieval path returns `RetrievalResult` objects.
- These interfaces should not be force-merged in a single cleanup step.
- `app/api.py`, `scripts/chat_cli.py`, and `rag/pipelines/chat_pipeline.py` still depend on the legacy path.
- Old evaluation scripts are still valid and should remain available, but they should be documented as legacy/general evaluators.
- `rag/modules/generation/*` exists but currently looks unwired; it should be documented, not deleted.

## Recommended Safe Cleanup Direction

1. Add documentation first.
2. Keep legacy runtime files as wrappers or compatibility layers.
3. Move duplicated evaluation metrics into a shared utility.
4. Reuse modular helpers from legacy wrappers where safe.
5. Avoid aggressive deletion in the first pass.
