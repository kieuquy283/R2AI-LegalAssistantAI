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

## Pipeline Optimizations (2026-06-26)

### Data Cleanup
- Xoá 45 "VĂN BẢN NÀY TRÙNG" duplicate points
- Fix 523 anomalous doc_numbers (leading quotes, "Khongso", "Số:", "Thông tư" prefix)
- Script: `scripts/clean_anomalous_doc_numbers.py`

### BM25 Optimizations
- k1, b → env vars `R2AI_BM25_K1`/`R2AI_BM25_B` (default 1.5/0.75)
- Bigrams optional flag `R2AI_BM25_BIGRAMS=true/false`
- Tokenizer preserves legal refs as single tokens (`45/2026/NĐ-CP`, `48-L/CTN`)
- Cache lock with portalocker (shared read/exclusive write)

### LEGAL_REF_PATTERN Updated
- `qdrant_retriever.py`, `legal_exact_search.py`, `run_full.py`, `reranker.py`
- Support old format `48-L/CTN` (no `/` prefix)

### RRF Improvements
- `rrf_sum * 10.0` → proper min-max normalization to [0,1]
- Global min-max norm after both RRF and linear fusion paths
- `CURRENT_YEAR` hardcoded → `datetime.now().year`
- `RRF_K` → env var `R2AI_RRF_K` (default 60)

### Heuristic Reranker Refactor
- 12+ if-else topic penalties → data-driven `_TOPIC_PENALTIES`/`_TOPIC_TITLE_PENALTIES` dicts
- Min thresholds → env vars `R2AI_HEURISTIC_MIN_FINAL/LEXICAL/TITLE`
- Difficulty limits → env vars `R2AI_DIFF_EASY_*`, `R2AI_DIFF_MID_*`, `R2AI_DIFF_HARD_*`, `R2AI_DIFF_VERYHARD_*`

### API Cascade
- Circuit breaker (3 failures/60s → 5min cooldown)
- Truncation 512 → 2048 (`R2AI_API_TRUNCATION`)
- Blend weight → env var `R2AI_API_WEIGHT` (default 0.5)
- Dynamic filter: removed `or s >= 0.05` (was no-op)
- Threshold: dùng 50% thay vì bypass hoàn toàn khi API-scored

### max_docs Cap
- very_hard 5→4, `select_dynamic_contexts` enforces doc_count for article-level items (max 4 distinct docs)

### Task 1: LLM Query Rewriting (2026-06-26)
- `_rewrite_query_llm()` in `query_expander.py` uses LLM to rewrite mid/hard questions
- Toggle: `R2AI_USE_LLM_QUERY_REWRITE=true/false`
- Skip easy questions to save cost/latency

### Task 2: Semantic Cache (2026-06-26)
- File: `src/qa_pipeline.py`
- Stores query→result in pickle cache (`data/cache/semantic_cache.pkl`)
- Cosine similarity threshold `R2AI_CACHE_SIM_THRESHOLD` (default 0.95)
- TTL `R2AI_CACHE_TTL` (default 3600s / 1 hour)
- Auto-prunes expired entries, flushes every 50 entries

### Task 3: Structured Output (2026-06-26)
- File: `src/generation/answer_generator.py`
- New `_generate_structured()` method requests JSON response with 4-section schema
- Parse JSON or fallback to regular free-text generation
- Toggle: `R2AI_USE_STRUCTURED_OUTPUT=true/false`
- Generation mode tracked: `llm_structured` / `llm` / `template`

### Task 4: Multi-Query Retrieval (2026-06-26)
- File: `src/retrieval/retrieval_pipeline.py`
- `_generate_multi_queries()` uses LLM (t=0.3) to create 3 variants for mid/hard questions
- Each variant is independently retrieved, results deduplicated and merged via fusion
- Toggle: `R2AI_USE_MULTI_QUERY=true/false`

### Task 5: Iterative Retrieval / CRAG (2026-06-26)
- File: `src/retrieval/retrieval_pipeline.py`
- `_crag_refine_query()` checks if `final_contexts < R2AI_CRAG_MIN_CONTEXTS` (default 2)
- If low, generates broader/fewer-specificity query → re-retrieve → merge results
- Only runs when improvement is detected (new result has more contexts)
- Toggle: `R2AI_USE_CRAG=true/false`

### Task 6: Adaptive Retrieval Depth (2026-06-26)
- File: `src/retrieval/retrieval_pipeline.py`
- `_estimate_difficulty()` drives `adapted_rerank_n` (cuts fusion output before reranker)
- Scale: easy=0.5, mid=1.0, hard=1.5, very_hard=2.0

### Task 7: Parallel Retrieval (2026-06-26)
- File: `src/retrieval/retrieval_pipeline.py`
- Uses `ThreadPoolExecutor` with `max_workers=min(n_queries, 4)`
- Runs dense + sparse + exact for each query variant concurrently
- Toggle: `R2AI_USE_PARALLEL_RETRIEVAL=true/false`

### Task 8: Distilled Cross-Encoder Config (2026-06-26)
- Already supported via env var `HYBRID_RERANKER_MODEL` (default `BAAI/bge-reranker-v2-m3`)
- Model, batch size, max length all configurable through env vars

### Task 9: Structured Monitoring Log (2026-06-26)
- File: `src/qa_pipeline.py`
- JSON log entry at end of each `answer()` call with question, route, elapsed_s, n_contexts, n_docs, n_articles, low_confidence, gen_mode
- Compatible with log aggregation tools

### Task 10: Regression Testing Script (2026-06-26)
- File: `scripts/eval_regression.py`
- `python scripts/eval_regression.py --questions-file <path> --output report.json [--sample N] [--verbose]`
- Reports: coverage, low_confidence count, CRAG usage, avg contexts/docs/time
- Seed=42 for reproducible sampling

### F2 Optimization Phase 2 (2026-06-26)

**Qdrant Dense Search** (`qdrant_retriever.py`):
- Thêm `score_threshold=0.25` vào cả `client.search()` và `query_points()` — lọc nhiễu sớm
- Tăng `CANDIDATE_K_ARTICLES=150` → `CANDIDATE_K_ARTICLES=250`

### F2 Recall Optimization (2026-06-26)

**Domain Adjustment Fix** (`qdrant_retriever.py`):
- Không còn overwrite `dense_score` và `final_score` — dùng `domain_adjusted_score` riêng
- Fusion thấy scores gốc, không bị distort bởi labor boost/penalty

**Multi-Query Dense Fix** (`retrieval_pipeline.py`):
- Sequential path: tất cả query variants đều chạy dense search (trước đây chỉ first variant)

**Loại bỏ redundant normalizations**:
- `qdrant_retriever.py`: bỏ `_minmax_normalize_dense_scores()` — giữ raw cosine scores
- `hybrid_fusion.py`: bỏ global min-max normalization cuối fusion

**Weight tuning** (`hybrid_fusion.py`):
- `citation_match`: 0.02 → **0.05**
- `temporal_boost`: ×1.0 → **×1.2**

**Threshold giảm** (`.env`):
- `ABSOLUTE_SCORE_THRESHOLD`: 0.15 → **0.10**
- `RELATIVE_SCORE_THRESHOLD`: 0.35 → **0.25**

**Qdrant Dynamic HNSW** (`qdrant_retriever.py` + `retrieval_pipeline.py`):
- `search()` nhận thêm `difficulty` param
- HNSW ef_search và score_threshold thay đổi theo difficulty:
  | Difficulty | ef_search | score_threshold | Lý do |
  |---|---|---|---|
  | easy | 128 | 0.30 | Nhanh, ít noise |
  | mid | 128 | 0.25 | Cân bằng |
  | hard | 192 | 0.20 | Recall cao hơn |
  | very_hard | 256 | 0.15 | Recall tối đa |
- Limit tăng lên 2000 — HNSW tự động dừng theo threshold

**Fusion Weights** (`hybrid_fusion.py`):
- `lexical_overlap` coefficient: 0.05 → **0.08** (từ khoá trùng quan trọng hơn)
- `wrong_domain_penalty` multiplier: 1.0 → **1.5** (phạt domain sai mạnh hơn)
- Temporal boost: 0.03 → **0.10** cho đúng năm, 0.08 cho ≤2 năm

**Article Evidence Aggregation** (`hybrid_fusion.py`):
- Article có ≥3 chunks chất lượng cao → boost 1.2x

**BM25 Bigrams Dynamic** (`retrieval_pipeline.py`):
- `R2AI_BM25_BIGRAMS=true` cho mid/hard, false cho easy

**Query Classifier** (`query_classifier.py` — new file):
- Rule-based classification: muc_phat, thu_tuc, dinh_nghia, so_sanh, co_so_hieu
- Inject type-specific keywords vào expanded_query

**Env Vars**:
- `CANDIDATE_K_ARTICLES=250`, `CANDIDATE_K_SPARSE=200`, `CANDIDATE_K_TITLE=30`
- `RERANK_TOP_N=250`, `HYBRID_RERANKER_HEURISTIC_TOP_K=25`
- `ABSOLUTE_SCORE_THRESHOLD=0.15`, `RELATIVE_SCORE_THRESHOLD=0.35`

### Full 2000 Run (previous)

### Phase 3: Domain Assignment from HF Dataset (2026-06-26)

**Source**: `th1nhng0/vietnamese-legal-documents` `legacy/metadata` (518k docs)
- `document_number` (so_ky_hieu) → `legal_sectors` (English)
- Mapped 240,299 unique doc_numbers → 8 taxonomy domains via `SECTOR_TO_DOMAIN`

**Scripts**:
- `scripts/build_domain_map.py` — Load HF dataset, build `doc_number → domain` mapping
- `scripts/update_qdrant_domain.py` — Scroll 149,707 Qdrant points, overwrite payload with `domain` field

**Coverage**:
- 93.4% HF-mapped (139,883 points có doc_number match)
- 6.6% rule-based (9,824 từ doc_title keywords)
- 41% default business_law (61,357 — legitimate business docs)
- Domain distribution: business_law 41% | admin_penalty 23.6% | tax 11% | investment 10.1% | land 8.7% | labor 4% | social_insurance 1.2% | ip 0.4%

**Code enabled** (`qdrant_retriever.py`):
- `_allowed_domain()` — giờ filter theo preferred_domains (substring match)
- `_query_collection()` — Qdrant-level filter với `MatchValue` trên domain field
- `_make_candidate()` — dùng `payload.domain` thay vì `tag_1`/`tag_2` (đã có sẵn)

**Tác động**: Mở khóa toàn bộ domain-aware infra:
- `domain_match` / `wrong_domain_penalty` trong fusion → có tác dụng
- `apply_domain_adjustment` (labor boost) → chạy được
- `preferred_domains` routing → Qdrant filter được domain

### Phase 2: API Weight Sweep (2026-06-26)

**Kết quả sweep 20 mẫu**:
| Weight | Coverage | Avg Ctxs | Avg Docs | Time | Ghi chú |
|--------|----------|----------|----------|------|---------|
| 0.0 | 100% | 3.95 | 2.40 | 9.94s | Heuristic-only, nhanh nhất |
| 0.3 | 100% | 4.30 | 2.25 | 11.54s | |
| 0.5 | 100% | 4.40 | 2.30 | 11.73s | Nhiều contexts nhất |
| 0.7 | 100% | 5.00* | 5.00* | 5.79s* | Anomaly (sample issue) |
| 1.0 | 100% | 4.00 | 2.30 | 11.63s | API-only |

**Decision**: Giữ R2AI_API_WEIGHT=0.5 — all weights cho 100% coverage, 0.5 cho nhiều contexts nhất.
Có thể tắt API hoàn toàn (weight=0.0) để tiết kiệm $ + ~1.8s/query nếu cần.


### Phase 1 Fixes (2026-06-26)

**eval_regression.py fixed**:
- Added load_dotenv() before importing LegalQAPipeline — testing now uses correct .env config (legal_parquet_v2, proper thresholds)

**Keyword injection completed** (etrieval_pipeline.py):
- Added so_sanh: [so sánh, khác biệt, phân biệt, đối chiếu]
- Added co_so_hieu: [số hiệu văn bản, văn bản số, điều khoản]
- Covers all 5 query types now

**CRAG quality gate fixed** (etrieval_pipeline.py):
- Easy questions (min_ctx=-1) always skip CRAG regardless of best_score
- Avoids unnecessary LLM call + re-retrieval on simple questions

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
| `R2AI_USE_LLM_QUERY_REWRITE=false` | Use LLM to rewrite query for mid/hard questions |
| `R2AI_USE_SEMANTIC_CACHE=true` | Enable/disable semantic result cache |
| `R2AI_USE_STRUCTURED_OUTPUT=true` | Generate answer as JSON with 4 fields |
| `R2AI_USE_MULTI_QUERY=true` | Generate 3 query variants for better recall |
| `R2AI_USE_CRAG=true` | Re-retrieve with broader query if too few contexts |
| `R2AI_USE_PARALLEL_RETRIEVAL=true` | Run retrievers concurrently per query variant |
| `R2AI_CACHE_SIM_THRESHOLD=0.95` | Cosine similarity threshold for cache hit |
| `R2AI_CACHE_TTL=3600` | Cache TTL in seconds (default 1 hour) |
| `R2AI_CRAG_MIN_CONTEXTS=2` | Min contexts before CRAG triggers refinement |
| `R2AI_BM25_K1=1.5` | BM25 k1 parameter |
| `R2AI_BM25_B=0.75` | BM25 b parameter |
| `R2AI_BM25_BIGRAMS=false` | Enable bigram tokenization for BM25 |
| `R2AI_RRF_K=60` | RRF ranking constant |
| `R2AI_API_WEIGHT=0.5` | Blend weight between API and heuristic scores |
| `R2AI_API_TRUNCATION=2048` | Max text length sent to API reranker |
| `R2AI_HEURISTIC_MIN_FINAL=0.05` | Heuristic min final score threshold |
| `R2AI_HEURISTIC_MIN_LEXICAL=0.05` | Heuristic min lexical overlap threshold |
| `R2AI_HEURISTIC_MIN_TITLE=0.1` | Heuristic min title overlap threshold |
| `R2AI_DIFF_EASY_DOCS=2` | Max docs for easy questions |
| `R2AI_DIFF_EASY_ARTS=2` | Max articles for easy questions |
| `R2AI_DIFF_EASY_CTX=3` | Max contexts for easy questions |
| `R2AI_DIFF_HARD_DOCS=4` | Max docs for hard questions |
| `R2AI_DIFF_HARD_ARTS=10` | Max articles for hard questions |
| `R2AI_DIFF_HARD_CTX=12` | Max contexts for hard questions |
| `R2AI_DIFF_VERYHARD_DOCS=4` | Max docs for very hard questions |
| `R2AI_DIFF_VERYHARD_ARTS=12` | Max articles for very hard questions |
| `R2AI_DIFF_VERYHARD_CTX=12` | Max contexts for very hard questions |
| `CANDIDATE_K_ARTICLES=250` | Qdrant article candidates |
| `CANDIDATE_K_SPARSE=200` | BM25 candidates |
| `CANDIDATE_K_TITLE=30` | Exact search candidates |
| `RERANK_TOP_N=250` | Max candidates into reranker |
| `HYBRID_RERANKER_HEURISTIC_TOP_K=25` | Heuristic filter to top-K |
| `ABSOLUTE_SCORE_THRESHOLD=0.15` | Min absolute score after fusion |
| `RELATIVE_SCORE_THRESHOLD=0.35` | Min relative score vs best |
| `QDRANT_SCORE_THRESHOLD=0.25` | Qdrant dense search min score |

## Performance Benchmarks (updated)

| Scenario | Time | Notes |
|----------|------|-------|
| Pipeline init | ~25s | Embedding model + BM25 preload (2-4s) |
| Query with cross-encoder | ~16s | Too slow for production |
| Query heuristic-only | ~1s | Default without multi-query/CRAG |
| Query with API reranker | ~3s | SiliconFlow API |
| Multi-query (3 variants) | ~2-3x base | Parallel retrieval minimizes overhead |
| CRAG re-retrieval | ~+0.5s | Only triggers when contexts < min threshold |
| Full 2000 eval (heuristic) | 22.8 min | ~0.68s/q (pre-optimizations) |

## Critical Files

- `src/generation/prompt_builder.py` — prompt optimization
- `src/generation/answer_generator.py` — answer generation with structured JSON output
- `src/retrieval/query_expander.py` — query expansion (keyword + LLM rewrite)
- `src/retrieval/retrieval_pipeline.py` — retrieval pipeline with routing, multi-query, CRAG, parallel retrieval
- `src/retrieval/hybrid_fusion.py` — RRF min-max norm, difficulty-based selection, global norm
- `src/retrieval/hybrid_reranker.py` — API circuit breaker, truncation, blend weight, cross-encoder config
- `src/retrieval/reranker.py` — data-driven topic penalties, heuristic threshold env vars
- `src/retrieval/query_classifier.py` — query type classification for adaptive strategy
- `src/retrieval/bm25_retriever.py` — env-var BM25 (k1/b), cache lock, combined text fields
- `src/retrieval/legal_exact_search.py` — LEGAL_REF_PATTERN updated
- `src/retrieval/qdrant_retriever.py` — LEGAL_REF_PATTERN updated
- `src/qa_pipeline.py` — main pipeline entry point with semantic cache + monitoring
- `scripts/clean_anomalous_doc_numbers.py` — Qdrant doc_number cleanup
- `scripts/run_full.py` — full 2000 runner, updated _DOC_RE pattern
- `scripts/eval_regression.py` — regression testing script
- `data/processed/labeled_dataset_local.jsonl` — 1,995 labeled Q&A with `tu_khoa_phap_ly`
- `data/evaluation/r2ai_stage1_questions.jsonl` — 2,000 eval questions
- `data/processed/submission_parquet.json` — final output (2000 entries, pre-optimizations)
- `data/cache/bm25_cache.pkl` + `data/cache/chunks.db` — BM25 cache
