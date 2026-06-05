# R2AI Legal AI Assistant

## Project purpose

This repository refactors the existing Multi-turnRAG codebase into a legal-assistant workspace for the R2AI 2026 competition. The refactor keeps the current Python stack, retrieval modules, and indexing flow, while organizing the project around legal retrieval, grounded answer generation, evaluation, and submission.

## Architecture overview

- `rag/`: existing core runtime kept as the compatibility layer
- `src/legal_rag/`: stable package surface for new legal-assistant development
- `configs/`: retrieval, generation, and evaluation configuration
- `data/raw/`: raw datasets and sample questions
- `data/processed/`: normalized metadata, corpus, and gold annotations
- `data/indexes/`: FAISS and related index artifacts
- `data/submissions/`: `results.json` outputs and evaluation reports
- `legacy/`: demo UI/API and historical assets kept outside the active pipeline

## Setup

```bash
python -m venv rag_env
rag_env\Scripts\activate
pip install -r requirements.txt
```

Create a local `.env` from `.env.example` and adjust model paths if needed.

## Build corpus

Normalize the legal dataset into retrieval/evaluation assets:

```bash
python -m scripts.prepare_legal_dataset \
  --input-path data/Legal_Dataset_V1.json \
  --corpus-output data/legal_corpus_chunks.json \
  --eval-output data/multiturn_evaluation_legal.json
```

## Build index

Build an index from the normalized corpus:

```bash
python -m scripts.build_index \
  --mode from_json \
  --corpus-json data/legal_corpus_chunks.json \
  --index-dir data/indexes/default
```

## Run retrieval

Run the current hybrid legal retrieval evaluation path:

```bash
python -m scripts.evaluate_model_3_hybrid \
  --eval-path data/multiturn_evaluation_legal.json \
  --index-dir data/indexes/default \
  --corpus-path data/legal_corpus_chunks.json \
  --top-k 5 \
  --output-path data/submissions/retrieval_eval.json
```

## Generate submission

Task 2 adds the dedicated legal submission generator:

```bash
python scripts/generate_submission.py \
  --input data/raw/sample_questions.json \
  --output data/submissions/results.json \
  --config configs/retrieval.yaml
```

## Validate submission

Task 2 adds the dedicated submission validator:

```bash
python scripts/validate_submission.py \
  --input data/submissions/results.json
```

## Evaluate locally

Task 2 adds article-level Precision / Recall / F2 evaluation:

```bash
python scripts/evaluate_submission.py \
  --pred data/submissions/results.json \
  --gold data/processed/sample_gold.json \
  --output data/submissions/eval_report.json
```

## Team workflow

1. Keep active legal pipeline code in `src/legal_rag/` or the retained `rag/` core.
2. Move unused demo or historical surfaces into `legacy/` instead of deleting them immediately.
3. Update `CHANGELOG.md` after each task-sized change.
4. Prefer package imports such as `from legal_rag.retrieval.hybrid import HybridRetriever`.

## Source Registry

`src/ingestion/source_registry.py` manages offline crawl sources declared in `data/sources/sources.yaml`.

- Core domain: `business_law`
- Satellite domains: `investment_law`, `tax_law`, `labor_law`, `social_insurance`, `administrative_penalty`
- Do not run live crawling during evaluation
- Do not bypass login, paywall, or captcha protections

## Coding conventions

- Preserve the current Python stack and avoid framework migrations.
- Do not hard-code secrets or API keys.
- Keep retrieval, generation, evaluation, and submission modules separated.
- Add focused tests for any new legal metadata, aggregation, evaluation, or submission logic.
