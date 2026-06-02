## Current Status

- Repository already contains evaluation scripts for Model 1 through Model 7.
- Legal evaluation dataset is present at `data/multiturn_evaluation_legal.json`.
- Legal retrieval corpus is present at `data/legal_corpus_chunks.json`.
- Legal index directory is expected at `indexes/legal`.
- Model 1 has already been validated successfully with `top_k=5`.
- Model 8 HyDE evaluation script is not present yet and will need to be implemented.

## Commands To Run Model 3-7

### Basic Checks

```powershell
python -m compileall app rag scripts
python -m scripts.evaluate_model_3_hybrid --help
python -m scripts.evaluate_model_4_hybrid_rerank --help
python -m scripts.evaluate_model_5_hybrid_history --help
python -m scripts.evaluate_model_6_multi_query_hybrid --help
python -m scripts.evaluate_model_7_full_pipeline --help
```

### Model 3

```powershell
python -m scripts.evaluate_model_3_hybrid `
  --eval-path data/multiturn_evaluation_legal.json `
  --index-dir indexes/legal `
  --corpus-path data/legal_corpus_chunks.json `
  --top-k 5 `
  --output-path logs/eval_runs/model_3_hybrid_legal_top5.json
```

### Model 4

```powershell
python -m scripts.evaluate_model_4_hybrid_rerank `
  --eval-path data/multiturn_evaluation_legal.json `
  --index-dir indexes/legal `
  --corpus-path data/legal_corpus_chunks.json `
  --top-k 5 `
  --candidate-k 30 `
  --output-path logs/eval_runs/model_4_hybrid_rerank_legal_top5.json
```

### Model 5

```powershell
python -m scripts.evaluate_model_5_hybrid_history `
  --eval-path data/multiturn_evaluation_legal.json `
  --index-dir indexes/legal `
  --corpus-path data/legal_corpus_chunks.json `
  --top-k 5 `
  --history-top-k 4 `
  --output-path logs/eval_runs/model_5_hybrid_history_legal_top5.json
```

### Model 6

```powershell
python -m scripts.evaluate_model_6_multi_query_hybrid `
  --eval-path data/multiturn_evaluation_legal.json `
  --index-dir indexes/legal `
  --corpus-path data/legal_corpus_chunks.json `
  --top-k 5 `
  --history-top-k 4 `
  --num-queries 4 `
  --output-path logs/eval_runs/model_6_multi_query_hybrid_legal_top5.json
```

### Model 7

```powershell
python -m scripts.evaluate_model_7_full_pipeline `
  --eval-path data/multiturn_evaluation_legal.json `
  --index-dir indexes/legal `
  --corpus-path data/legal_corpus_chunks.json `
  --top-k 5 `
  --candidate-k 40 `
  --history-top-k 4 `
  --num-queries 4 `
  --output-path logs/eval_runs/model_7_full_pipeline_legal_top5.json
```

## Implementation Plan For Model 8 HyDE

1. Add `rag/modules/query_rewriting/hyde.py`.
2. Reuse existing LLM client utilities from `rag/generation/llm_client.py`.
3. Implement HyDE generation that accepts `rewritten_query` and optional `selected_history`.
4. Use a Vietnamese legal HyDE prompt that returns only hypothetical retrieval text.
5. Fall back to `rewritten_query` if HyDE generation fails or returns empty text.
6. Add `scripts/evaluate_model_8_hyde.py`.
7. Reuse existing helpers and flow from Model 7:
   - history normalization
   - hybrid history selection with recent fallback
   - query rewriting with fallback
   - hybrid retrieval
   - candidate deduplication by `cid`
   - reranking with fallback
8. Combine `rewritten_query` and HyDE query for retrieval when `include_original_query` is enabled.
9. Fuse retrieval results and evaluate only retrieval metrics against `ground_truth_cids`.
10. Ensure no answer generation pipeline is called.

## Validation Checklist

- `python -m compileall app rag scripts` passes.
- Model 3 through Model 8 support `--help`.
- Output JSON exists for every run in `logs/eval_runs`.
- Every output JSON contains metrics and per-sample results.
- Final retrieval comparison uses `retrieved_cids` or `retrieved_cids_after_rerank` against `ground_truth_cids`.
- Model 8 preserves `cid`, `chunk_id`, and `doc_id` metadata.
- No API or UI files are modified in a breaking way.
- No evaluation script triggers final answer generation.

## Expected Output Files

- `logs/eval_runs/model_3_hybrid_legal_top5.json`
- `logs/eval_runs/model_4_hybrid_rerank_legal_top5.json`
- `logs/eval_runs/model_5_hybrid_history_legal_top5.json`
- `logs/eval_runs/model_6_multi_query_hybrid_legal_top5.json`
- `logs/eval_runs/model_7_full_pipeline_legal_top5.json`
- `logs/eval_runs/model_8_hyde_legal_top5.json`
- `logs/eval_runs/comparison_legal_top5.csv`
