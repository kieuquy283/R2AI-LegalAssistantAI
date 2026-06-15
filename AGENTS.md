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
- `data/sources/domain_taxonomy.json`: added `land_law` with keywords (đất đai, thuê đất, sử dụng đất, etc.)
- `data/sources/sources.yaml`: added `luatvietnam_land_law_search` source
- `src/ingestion/hf_legal_filter_rules.py`: added `land_law` rule group + `administrative_penalty` rule group

### 3. Data Collection Enhancement

New rule groups in HF filter for missing domains:
- `land_law`: Luật đất đai, thuê đất, quyền sử dụng đất
- `administrative_penalty`: xử phạt vi phạm hành chính, mức phạt

To collect more data:
```bash
# 1. Filter HF dataset with new rules
python -m src.ingestion.filter_hf_legal_dataset --output data/raw/hf_filtered_land_penalty.jsonl --limit 1000

# 2. Or crawl from LuatVietnam
python -m src.ingestion.source_registry
python -m src.ingestion.collect_urls --limit 20
python -m src.ingestion.crawl_documents --limit 20

# 3. Run full ingestion
python -m scripts.run_ingestion --skip-crawl
```

## Eval Quality Status

- **100 câu eval**: 30.9s/câu, 100% citations, 100% non-empty
- **Route distribution**: PARENT_CONTEXT 72%, SIMPLE_VECTOR 16%, CROSS_DOMAIN 9%, LEGAL_GRAPH 3%
- **Quality**: 55% excellent, 30% verbose/repetitive, 20% "not specified" answers
- **Metrics**: comprehensive_eval scores 0% because no ground truth in eval data

## Next Steps

1. **Run HF filter** with new rules to collect land/penalty docs
2. **Ingest** new docs into Qdrant (or FAISS)
3. **Re-run eval** with improved prompt to measure conciseness gain
4. **Add ground truth** to eval data for meaningful metrics

## Critical Files

- `src/generation/prompt_builder.py` — prompt optimization
- `src/ingestion/hf_legal_filter_rules.py` — data filtering rules
- `data/sources/domain_taxonomy.json` — domain definitions
- `data/sources/sources.yaml` — crawl sources
- `scripts/run_qdrant_eval_100.py` — eval runner
- `data/cache/bm25_cache.pkl` + `data/cache/chunks.db` — BM25 cache

## Environment

- Windows, NVIDIA MX350 (2GB VRAM, not used — CPU-only)
- PyTorch CPU-only
- Qdrant via Docker on `localhost:6333`
- HuggingFace cache: `D:\huggingface_cache`
- API reranker: SiliconFlow (`https://api.siliconflow.com/v1/rank`)
