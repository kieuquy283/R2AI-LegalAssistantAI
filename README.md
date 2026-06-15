# R2AI Legal Assistant — Vietnamese Legal RAG for SME

Hệ thống Retrieval-Augmented Generation (RAG) tư vấn pháp lý cho doanh nghiệp vừa và nhỏ (SME) tại Việt Nam. Hệ thống truy xuất và diễn giải văn bản pháp luật tiếng Việt theo cấu trúc pháp lý (Điều → Khoản → Điểm), trả lới câu hỏi có trích dẫn chính xác và hành động thực tế cho SME.

---

## Mục lục

1. [Tổng quan kiến trúc](#tổng-quan-kiến-trúc)
2. [Luồng dữ liệu 3-phase](#luồng-dữ-liệu-3-phase)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Cài đặt & Môi trường](#cài-đặt--môi-trường)
5. [Phase 1: HF Dataset Filter + Time Filtering](#phase-1-hf-dataset-filter--time-filtering)
6. [Phase 2: Chunking + Graph + Kaggle Embedding](#phase-2-chunking--graph--kaggle-embedding)
7. [Phase 3: Qdrant Upsert](#phase-3-qdrant-upsert)
8. [Retrieval & QA Pipeline](#retrieval--qa-pipeline)
9. [Hệ thống Đánh giá](#hệ-thống-đánh-giá)
10. [Quyết định thiết kế chính](#quyết-định-thiết-kế-chính)

---

## Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           R2AI Legal Assistant                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  User Query → Query Router → Hybrid Retrieval → Context Expansion → LLM    │
│                                ↓                    ↓                       │
│                         [Qdrant + BM25]      [Graph Traversal]              │
│                                ↓                    ↓                       │
│                         Dense + Sparse + Exact   Parent/Neighbor            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stack công nghệ

| Layer | Công nghệ |
|---|---|
| **Vector DB** | Qdrant (5 collections: docs, articles, chunks, context_chunks, edges) |
| **Embedding** | BAAI/bge-m3 (1024 dims, cosine similarity) |
| **Sparse Retrieval** | BM25 (SQLite + disk cache, ~11s warm load) |
| **Reranker** | Hybrid fusion (dense × sparse × exact) + SiliconFlow API fallback |
| **Infra** | Docker (Qdrant) |

---

## Luồng dữ liệu 3-phase

```
Phase 1: Ingestion + Expansion
    HF Dataset (th1nhng0/vietnamese-legal-documents)
    → Stream Parquet (metadata + content)
    → Domain keyword filter (7 rule groups)
    → Time-based filter (effective_date ≤ 2026-03-01, still valid)
    → Output: hf_filtered_2026.jsonl (17,012 docs)

Phase 2: Chunking + Graph + Kaggle Embedding
    hf_filtered_2026.jsonl
    → legal_chunker.py (Điều/Khoản/Điểm parsing)
    → export_chunks_for_kaggle.py
    → Kaggle T4 GPU: BAAI/bge-m3 embed
    → Output: legal_chunks_embedded.jsonl + legal_context_chunks_embedded.jsonl

Phase 3: Qdrant Upsert
    Embedded chunks + context_chunks + edges
    → 5 Qdrant collections with payload indexes
    → Ready for retrieval
```

---

## Cấu trúc thư mục

```text
rag_chatbot/
├── src/
│   ├── ingestion/              # Pipeline thu thập & xử lý dữ liệu
│   │   ├── filter_hf_legal_dataset.py      # Phase 1: HF filter + time filter
│   │   ├── hf_legal_filter_rules.py        # Domain rules + date extraction
│   │   ├── legal_chunker.py                # Phase 2: Graph-aware chunking
│   │   ├── legal_structure_parser.py       # Parse Điều/Khoản/Điểm
│   │   ├── bm25_builder.py                 # BM25 index builder
│   │   ├── qdrant_index_builder.py         # Qdrant index builder
│   │   ├── source_registry.py              # Crawl source registry
│   │   ├── crawl_documents.py              # LuatVietnam crawler
│   │   └── ...
│   ├── retrieval/              # Hệ thống truy xuất
│   │   ├── retrieval_pipeline.py           # Entry point, singleton
│   │   ├── query_router.py                 # Route detection (5 strategies)
│   │   ├── qdrant_retriever.py             # Dense retrieval
│   │   ├── bm25_retriever.py               # Sparse retrieval
│   │   ├── legal_exact_search.py           # Exact legal ref matching
│   │   ├── hybrid_fusion.py                # Dense + Sparse + Exact fusion
│   │   ├── hybrid_reranker.py              # Hybrid reranking
│   │   ├── context_expander.py             # Graph expansion (parent/neighbor)
│   │   └── ...
│   ├── generation/             # Sinh câu trả lới
│   │   ├── answer_generator.py             # LLM generation + fallback
│   │   ├── prompt_builder.py               # 4-section prompt (anti-repetition)
│   │   ├── llm_client.py                   # OpenRouter/GPT-4o-mini client
│   │   └── grounding_validator.py          # Verify citations in answer
│   ├── evaluation/             # Đánh giá
│   │   ├── comprehensive_evaluator.py      # Citation accuracy, coverage
│   │   ├── evaluate_qa.py                  # Batch eval runner
│   │   ├── eval_logger.py                  # Per-run JSONL logging
│   │   └── error_analyzer.py               # Error pattern analysis
│   ├── qa_pipeline.py          # End-to-end QA singleton
│   └── legal_rag/              # Legacy modules (retained)
│       ├── corpus/, retrieval/, generation/, submission/
├── scripts/                    # Standalone utilities
│   ├── export_chunks_for_kaggle.py         # Export Kaggle-ready format
│   ├── kaggle_embedding_job.py             # Embed on Kaggle T4
│   ├── upsert_docs_articles.py             # Upsert to Qdrant (5 collections)
│   ├── run_ingestion.py                    # Full ingestion orchestrator
│   └── run_qdrant_eval_100.py              # 100-question eval runner
├── data/
│   ├── raw/                    # Raw data
│   │   ├── hf_filtered_2026.jsonl          # 17,012 filtered legal docs
│   │   └── documents_manifest.jsonl
│   ├── processed/              # Processed artifacts
│   │   ├── chunks.jsonl                    # Article/clause/point chunks
│   │   ├── context_chunks.jsonl            # Article-level context chunks
│   │   ├── legal_edges.jsonl               # Graph edges (HAS_PARENT, etc.)
│   │   ├── legal_chunks_to_embed.jsonl     # Kaggle input (chunks)
│   │   └── legal_context_chunks_to_embed.jsonl  # Kaggle input (context)
│   ├── sources/                # Config
│   │   ├── domain_taxonomy.json            # Domain definitions
│   │   └── sources.yaml                    # Crawl source registry
│   ├── indexes/                # FAISS + BM25 artifacts
│   └── evaluation/             # Eval datasets
│       └── r2ai_stage1_questions.jsonl     # 2,000 eval questions
├── logs/                       # Eval logs & reports
├── tests/                      # Unit & smoke tests
├── rag/                        # Compatibility layer (config, vectorstore)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Cài đặt & Môi trường

### 1. Python environment

```bash
python -m venv rag_env
rag_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Qdrant (Docker)

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 3. Environment variables

Copy `.env.example` → `.env`:

```bash
# Retrieval backend
RETRIEVAL_BACKEND=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embedding (local CPU)
EMBEDDING_MODEL=BAAI/bge-m3

# Reranker
SILICONFLOW_API_KEY=sk-...

# Cache
HF_HOME=D:\huggingface_cache
```

---

## Phase 1: HF Dataset Filter + Time Filtering

### Mục tiêu
Lọc dataset `th1nhng0/vietnamese-legal-documents` (~156k docs) theo:
1. **Domain keywords** (7 rule groups: business_sme, tax, labor, ip, commerce, land_law, administrative_penalty)
2. **Time validity** (`effective_date ≤ cutoff` và vẫn còn hiệu lực)

### Chạy

```bash
python -m src.ingestion.filter_hf_legal_dataset \
  --cutoff-date 2026-03-01 \
  --output data/raw/hf_filtered_2026.jsonl \
  --chunks-output data/processed/legal_chunks_to_embed.jsonl
```

### Output

| Metric | Value |
|---|---|
| Scanned | 156,624 records |
| Matched domain | 18,083 |
| Time-excluded | 1,014 |
| Final deduped | **17,012** |

### Domain distribution

| Domain | Count | % |
|---|---|---|
| administrative_penalty | 8,891 | 52.3% |
| land_law | 4,482 | 26.3% |
| labor_bhxh_union | 1,244 | 7.3% |
| business_sme | 1,187 | 7.0% |
| commerce_procurement_customs | 735 | 4.3% |
| tax_invoice_accounting | 330 | 1.9% |
| intellectual_property | 143 | 0.8% |

### Logic lọc thờigian

```python
effective_date <= '2026-03-01' AND (
    expiry_date > '2026-03-01'
    OR expiry_date IS NULL
    OR status == 'Vẫn còn hiệu lực'
)
```

- `issued_date` được parse riêng biệt (không nhầm với `effective_date`)
- Regex fallback chỉ trên preamble (5000 chars đầu) để tránh match nhầm ngày trong nội dung

---

## Phase 2: Chunking + Graph + Kaggle Embedding

### Bước 2.1: Graph-aware Chunking

```bash
python -m src.ingestion.legal_chunker
```

**Cấu trúc chunk:**

```json
{
  "chunk_id": "doc_001_article_17_clause_1",
  "doc_id": "doc_001",
  "level": "clause",
  "article": "Điều 17",
  "clause": "Khoản 1",
  "point": null,
  "legal_path": "Luật Doanh nghiệp 2020 > Điều 17 > Khoản 1",
  "citation": "Luật Doanh nghiệp 2020, Điều 17, Khoản 1",
  "content": "Công ty TNHH...",
  "embedding_text": "Luật Doanh nghiệp 2020\nĐiều 17\nCông ty TNHH...",
  "parent_id": "doc_001_article_17",
  "context_chunk_id": "doc_001_article_17_context",
  "prev_chunk_id": "doc_001_article_16",
  "next_chunk_id": "doc_001_article_17_clause_2",
  "effective_date": "2021-01-01",
  "expiry_date": null,
  "status": "Vẫn còn hiệu lực"
}
```

**Context chunk (article level):**

```json
{
  "context_chunk_id": "doc_001_article_17_context",
  "level": "article",
  "article": "Điều 17",
  "child_chunk_ids": ["doc_001_article_17_clause_1", "doc_001_article_17_clause_2"],
  "effective_date": "2021-01-01",
  "expiry_date": null
}
```

### Bước 2.2: Export sang Kaggle format

```bash
python scripts/export_chunks_for_kaggle.py
```

Output 2 files:
- `data/processed/legal_chunks_to_embed.jsonl`
- `data/processed/legal_context_chunks_to_embed.jsonl`

Format mỗi dòng:
```json
{"id": "...", "text": "embedding_text", "metadata": {...}}
```

### Bước 2.3: Embed trên Kaggle T4

```python
# Kaggle notebook
!pip install sentence-transformers

!python scripts/kaggle_embedding_job.py \
  --input /kaggle/input/legal-chunks/legal_chunks_to_embed.jsonl \
  --output /kaggle/working/legal_chunks_embedded.jsonl \
  --model BAAI/bge-m3 \
  --batch-size 32

!python scripts/kaggle_embedding_job.py \
  --input /kaggle/input/legal-context/legal_context_chunks_to_embed.jsonl \
  --output /kaggle/working/legal_context_chunks_embedded.jsonl \
  --model BAAI/bge-m3 \
  --batch-size 32
```

**Tính năng:**
- L2 normalize vectors (cho cosine similarity)
- Streaming mode (`--streaming`) với checkpoint append để tránh Kaggle timeout
- Output: `{"id": "...", "vector": [0.01, ...], "metadata": {...}}`

---

## Phase 3: Qdrant Upsert

### Upsert toàn bộ graph

```bash
python scripts/upsert_docs_articles.py --type all --recreate
```

### Schema 5 collections

| Collection | Content | Vector | Payload Indexes |
|---|---|---|---|
| `legal_docs` | Document level | bge-m3 (1024d, Cosine) | `effective_date` (DATETIME), `expiry_date` (DATETIME), `status` (KEYWORD) |
| `legal_articles` | Article nodes | bge-m3 (1024d, Cosine) | `effective_date` (DATETIME), `expiry_date` (DATETIME), `status` (KEYWORD) |
| `legal_chunks` | Điều/Khoản/Điểm chunks | pre-computed bge-m3 | `parent_id`, `prev_chunk_id`, `next_chunk_id`, `context_chunk_id` (KEYWORD), + time fields |
| `legal_context_chunks` | Article context | pre-computed bge-m3 | `child_chunk_ids` (KEYWORD), + time fields |
| `legal_edges` | Graph edges | dummy `[1e-6]*1024` | `source_id`, `target_id`, `relation_type` (KEYWORD) |

### Edge types

| Relation | Description |
|---|---|
| `HAS_PARENT` | Chunk → Article/Clause parent |
| `PREV_CHUNK` | Chunk → Previous sequential chunk |
| `NEXT_CHUNK` | Chunk → Next sequential chunk |

### Upsert riêng lẻ

```bash
python scripts/upsert_docs_articles.py --type chunks --chunks-file ...
python scripts/upsert_docs_articles.py --type context --context-chunks-file ...
python scripts/upsert_docs_articles.py --type edges --edges-file ...
```

---

## Retrieval & QA Pipeline

### Architecture

```
User Question
    ↓
[Query Router] → detect domains → select route (5 strategies)
    ↓
[Retrieval Pipeline]
    ├── Qdrant Dense (vector search, domain-prefiltered)
    ├── BM25 Sparse (keyword search)
    └── Legal Exact (Điều/Khoản pattern matching)
    ↓
[Hybrid Fusion] → merge & score candidates
    ↓
[Hybrid Reranker] → final ranking
    ↓
[Context Expander] → graph traversal (parent/neighbor/cross-domain)
    ↓
[Answer Generator]
    ├── Prompt Builder (4-section anti-repetition prompt)
    ├── LLM (GPT-4o-mini)
    └── Grounding Validator (verify citations)
    ↓
Structured Answer + Citations
```

### 5 Retrieval Routes

| Route | Trigger | Strategy |
|---|---|---|
| `SIMPLE_VECTOR` | Simple factual query | Vector search only |
| `PARENT_CONTEXT` | Article/clause grounding | Vector + parent expansion |
| `LEGAL_GRAPH_CONTEXT` | Relationship/status query | Vector + graph traversal |
| `CROSS_DOMAIN_CONTEXT` | Multi-domain query | Vector + cross-domain edges |
| `MULTI_DOMAIN_COMPLEX` | Broad guidance request | All sources + neighbor expansion |

### Answer Format (4 phần bắt buộc)

```
1. Kết luận ngắn
   (1-2 câu tóm tắt)

2. Căn cứ pháp luật
   (Trích dẫn: Tên văn bản, Điều, Khoản, Điểm [context_index])

3. Phân tích áp dụng
   (Diễn giải tự nhiên, số liệu cụ thể)

4. Việc SME nên làm
   (Checklist 3-5 bước hành động cụ thể)
```

### Chạy QA

```bash
# Full pipeline
python -m src.qa_pipeline --question "Ai không được thành lập doanh nghiệp?"

# Từng thành phần
python -m src.retrieval.retrieval_pipeline --query "..."
python -m src.generation.answer_generator --query "..."
```

---

## Hệ thống Đánh giá

### Batch evaluation (100 câu)

```bash
python scripts/run_qdrant_eval_100.py
```

### Metrics

| Metric | Target | Current |
|---|---|---|
| Citation present rate | 100% | ✅ 100% |
| Answer non-empty rate | 100% | ✅ 100% |
| Avg latency | < 60s | ✅ 30.9s |
| Route: PARENT_CONTEXT | — | 72% |
| Route: SIMPLE_VECTOR | — | 16% |
| Route: CROSS_DOMAIN | — | 9% |
| Route: LEGAL_GRAPH | — | 3% |
| Quality: Excellent | > 50% | ✅ 55% |
| Quality: Verbose | < 30% | ⚠️ 30% |

### Comprehensive Evaluator

```python
from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator

# Citation accuracy: điều luật trong answer có xuất hiện trong context?
# Grounding check: answer có dựa trên context?
# Completeness: 4 phần bắt buộc có đầy đủ?
```

---

## Quyết định thiết kế chính

### 1. Time-based Filtering
- Mọi văn bản phải thỏa mãn: `effective_date ≤ cutoff` AND (`expiry_date > cutoff` OR `expiry_date IS NULL`)
- Payload indexes DATETIME trên `effective_date` và `expiry_date` cho pre-filtering tại Qdrant
- `issued_date` parse riêng biệt, không dùng làm `effective_date` proxy

### 2. Graph trong Qdrant
- Edges lưu trong collection riêng (`legal_edges`) với dummy vectors
- Traversal qua `source_id`/`target_id` KEYWORD indexes
- Chunks lưu graph refs (`parent_id`, `prev_chunk_id`, `next_chunk_id`) trong payload để expand nhanh

### 3. Kaggle Embedding Strategy
- Không embed local (CPU quá chậm với 17k docs)
- Export JSONL → Kaggle T4 → batch embed → download vectors → fast local upsert
- Streaming mode với checkpoint để tránh Kaggle timeout

### 4. Hybrid Retrieval (Qdrant backend)
- **Dense**: Qdrant vector search (bge-m3, cosine)
- **Sparse**: BM25 (SQLite-backed, disk cache)
- **Exact**: Regex matching Điều/Khoản/Điểm patterns
- **Fusion**: Weighted reciprocal rank fusion với domain boost

### 5. Anti-Hallucination
- Grounding validator kiểm tra citations trong answer
- Forbidden phrases detection ("context không cung cấp", "chưa đủ căn cứ" khi context có dữ liệu)
- Fallback answer 4-phần nếu LLM vi phạm format
- Không bịa điều luật, số hiệu văn bản

### 6. Prompt Engineering
- System prompt: 4 phần bắt buộc + anti-repetition rules
- User prompt: simplified, không duplicate system instructions
- Tối đa 300-400 từ, loại bỏ văn bản địa phương/quá cũ

---

## Tham khảo nhanh

### Các file quan trọng

| File | Mục đích |
|---|---|
| `src/ingestion/filter_hf_legal_dataset.py` | HF dataset filter + time filter |
| `src/ingestion/hf_legal_filter_rules.py` | Domain rules + date extraction |
| `src/ingestion/legal_chunker.py` | Graph-aware chunking |
| `scripts/export_chunks_for_kaggle.py` | Export Kaggle format |
| `scripts/kaggle_embedding_job.py` | Kaggle T4 embedding |
| `scripts/upsert_docs_articles.py` | Qdrant upsert (5 collections) |
| `src/retrieval/retrieval_pipeline.py` | Retrieval entry point |
| `src/retrieval/query_router.py` | Route detection |
| `src/retrieval/hybrid_fusion.py` | Dense + Sparse + Exact fusion |
| `src/generation/prompt_builder.py` | 4-section prompt |
| `src/qa_pipeline.py` | End-to-end QA |
| `rag/config/runtime.py` | Runtime configuration |

### Variables môi trường quan trọng

```bash
RETRIEVAL_BACKEND=qdrant          # qdrant | faiss
QDRANT_HOST=localhost
QDRANT_PORT=6333
CANDIDATE_K_CHUNKS=150            # Top-k chunks
CANDIDATE_K_SPARSE=150            # BM25 top-k
RERANK_TOP_N=50                   # Reranker input
MAX_CONTEXTS=8                    # Final contexts to LLM
CITATION_SCORE_THRESHOLD=0.50     # Min score for citation
```

---

## License

Internal use for R2AI Legal Assistant project.
