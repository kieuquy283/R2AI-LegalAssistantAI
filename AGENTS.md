# R2AI Legal AI Assistant — Agent Notes

## Project Context

Vietnamese legal RAG chatbot for SME (small/medium enterprise) consulting.

## Current Architecture

- **Retrieval**: Qdrant (3 collections: legal_chunks, legal_docs, legal_articles)
- **Embedding**: BAAI/bge-m3 (local CPU, 1024 dims)
- **Reranker**: API (SiliconFlow) + heuristic fallback
- **LLM**: GPT-4o-mini via API (OpenRouter or direct)
- **BM25**: SQLite + disk cache for fast warm load (~11s)
- **Pipeline**: Singleton with pre-loaded BM25

## Dataset Stats (after JSONL merge + enterprise/labor 2026-06-25)

| Metric | Before | After |
|---|---|---|
| Vectors (npy) | 122,068 | **171,330** |
| plus enterprise/labor (Kaggle GPU) | — | +1,049 |
| **Qdrant total** | 122,068 | **172,379** |
| Docs | 17,003 | **22,113** |
| Added from JSONL | — | 48,505 arts / 5,110 docs |
| Enterprise/labor articles | — | 910 + 155 = 1,049 |
| Expired removed | 5,333 | 5,333 |

## Environment

- Windows, NVIDIA MX350 (2GB VRAM, not used — CPU-only)
- PyTorch CPU-only
- Qdrant via Docker on `localhost:6333`
- HuggingFace cache: `D:\huggingface_cache`
- API reranker: SiliconFlow (`https://api.siliconflow.com/v1/rank`)

## Key Changes (2024-06-13)

### 1. Prompt Optimization (Anti-Repetition)

File: `src/generation/prompt_builder.py`

Added to system prompt:
- `TRÁNH LẶP LẶI: mỗi nội dung chỉ nêu một lần`
- `SÚC TÍCH: câu trả lời tối đa 300-400 từ`
- `KHÔNG dùng bullet point trùng lặp giữa các phần`

Simplified user prompt to reduce duplication with system prompt.

### 2. New Domain: Land Law

Files:
- `data/sources/domain_taxonomy.json`: added `land_law` domain
- `data/sources/sources.yaml`: added `luatvietnam_land_law_search` source
- `src/ingestion/hf_legal_filter_rules.py`: added `land_law` + `administrative_penalty` rules

## Retrieval Optimization (2026-06-26)

### Cross-Encoder Disabled
- `BAAI/bge-reranker-v2-m3` cross-encoder took ~15s/query on CPU → disabled
- Set `HYBRID_RERANKER_ENABLE_CROSS_ENCODER=false`

### Best Config Found (100-sample eval, seed=42)
| Config | Overall | Coverage | Format | National | Avg Docs | Time |
|--------|---------|----------|--------|----------|----------|------|
| heuristic-only | **88.2** | 100% | 90.1% | 92.4% | 3.02 | ~1s/q |
| **heuristic+kwexpand** | **89.0** | 100% | 92.4% | 93.4% | 3.02 | ~1s/q |
| api+kwexpand | 84.2 | 100% | 89.0% | 84.0% | 2.37 | ~3s/q |
| api-only | 85.2 | 100% | 89.0% | 87.0% | 2.46 | ~3s/q |

**Best**: heuristic+kwexpand (heuristic-only + keyword expansion from labeled_dataset)

### Full 2000 Run
- **Config**: `HYBRID_RERANKER_ENABLE_CROSS_ENCODER=false`, `R2AI_USE_RRF=true`, `HYBRID_RERANKER_API_ENABLED=false`, `R2AI_USE_KEYWORD_EXPANSION=true`
- **Time**: 22.8 minutes (2000 queries, ~0.68s/q)
- **Coverage**: **100%** (0 empty)
- **Format rate**: **100%** (0 bad entries) — post-filtered 437 malformed doc_keys (provincial docs without legal number), re-ran pipeline for affected entries
- **National rate**: 93.3%
- **Avg docs**: 1.96 (1.96 when found)
- **Overall**: **~85** (slight drop due to filtering provincial docs)
- **Output**: `data/processed/submission_parquet.json`

## Feature Flags (env vars)

| Flag | Effect |
|---|---|
| `R2AI_USE_KEYWORD_EXPANSION=true` | Use `labeled_dataset_local.jsonl` keywords for query expansion |
| `R2AI_FORCE_SIMPLE_ROUTE=true` | Disable adaptive routing, force SIMPLE_VECTOR |
| `R2AI_DISABLE_ANSWER=true` | Skip answer generation, output empty answer field |
| `R2AI_USE_RRF=true` | Reciprocal Rank Fusion thay vì weighted linear sum |
| `HYBRID_RERANKER_SKIP_HEURISTIC=true` | Skip heuristic stage, send all candidates to API reranker |
| `R2AI_RETRIEVAL_SKIP_EXPANSION=true` | Tắt hoàn toàn query expansion |
| `HYBRID_RERANKER_ENABLE_CROSS_ENCODER=false` | Tắt cross-encoder model (rất chậm trên CPU) |
| `HYBRID_RERANKER_API_ENABLED=false` | Tắt API reranker (SiliconFlow) |
| `HYBRID_RERANKER_HEURISTIC_TOP_K=15` | Số candidate heuristic filter (default: 15) |

## Performance Benchmarks

| Scenario | Time | Notes |
|----------|------|-------|
| Pipeline init | ~25s | Embedding model + BM25 preload (2-4s) |
| Query with cross-encoder | ~16s | Too slow for production |
| Query heuristic-only | ~1s | Default (recommended) |
| Query with API reranker | ~3s | SiliconFlow API |
| Full 2000 eval (heuristic) | 22.8 min | ~0.68s/q |

## Critical Files

- `src/generation/prompt_builder.py` — prompt optimization
- `src/retrieval/query_expander.py` — query expansion (keyword-based from labeled_dataset_local.jsonl)
- `src/retrieval/retrieval_pipeline.py` — retrieval pipeline with routing
- `src/qa_pipeline.py` — main pipeline entry point
- `data/processed/labeled_dataset_local.jsonl` — 1,995 labeled Q&A with `tu_khoa_phap_ly`
- `data/evaluation/r2ai_stage1_questions.jsonl` — 2,000 eval questions
- `data/processed/submission_parquet.json` — final output (2000 entries)
- `data/cache/bm25_cache.pkl` + `data/cache/chunks.db` — BM25 cache
