# R2AI Legal Assistant

Hệ thống RAG pháp lý cho bài toán tư vấn doanh nghiệp/SME. Repo này hiện có đủ:

- offline ingestion pipeline
- FAISS + BM25 retrieval
- query routing
- context expansion theo parent / legal graph / cross-domain
- reranking
- answer generation có citation
- evaluation logging và smoke evaluation

## Cấu trúc chính

```text
rag/                     Compatibility layer và vectorstore hiện có
src/ingestion/           Pipeline crawl, parse, clean, chunk, enrich, index
src/retrieval/           Hybrid retriever, router, expander, reranker
src/generation/          Answer generator
src/evaluation/          Eval logger và evaluation runner
src/qa_pipeline.py       End-to-end QA pipeline
src/legal_rag/           Các module legal_rag trước đó vẫn được giữ lại
data/raw/                URL, raw documents, manifest
data/processed/          Documents, chunks, legal nodes/edges, reports
data/indexes/            FAISS, BM25 metadata
logs/eval_runs/          Eval logs và summary
tests/                   Unit tests và pipeline smoke tests
```

## Cài đặt

```bash
python -m venv rag_env
rag_env\Scripts\activate
pip install -r requirements.txt
```

Nếu có `.env`, giữ theo `.env.example`. Repo không yêu cầu crawl live để chạy retrieval/QA/evaluation đã build sẵn.

## Dữ liệu đầu ra hiện có

Các artifact chính đã được build:

- `data/raw/documents_manifest.jsonl`
- `data/processed/documents.jsonl`
- `data/processed/cleaned_documents.jsonl`
- `data/processed/legal_nodes.jsonl`
- `data/processed/legal_edges.jsonl`
- `data/processed/chunks.jsonl`
- `data/processed/context_chunks.jsonl`
- `data/indexes/faiss.index`
- `data/indexes/chunk_metadata.json`
- `data/indexes/bm25_corpus.json`
- `data/indexes/bm25_metadata.json`

## Ingestion pipeline

Thứ tự pipeline hiện tại:

```text
source registry
-> collect URLs
-> crawl detail pages
-> save markdown
-> create manifest/documents
-> extract metadata
-> clean text
-> parse điều/khoản
-> create chunks
-> create context chunks
-> explicit refs / cross-domain edges
-> incremental update state
-> BM25 + FAISS index
-> sanity report
```

Các module chính:

- `src/ingestion/source_registry.py`
- `src/ingestion/collect_urls.py`
- `src/ingestion/crawl_documents.py`
- `src/ingestion/document_parser.py`
- `src/ingestion/text_cleaner.py`
- `src/ingestion/legal_structure_parser.py`
- `src/ingestion/legal_chunker.py`
- `src/ingestion/reference_enricher.py`
- `src/ingestion/bm25_builder.py`
- `src/ingestion/index_builder.py`
- `src/ingestion/incremental_update.py`
- `src/ingestion/sanity_report.py`

Ví dụ chạy từng bước:

```bash
python -m src.ingestion.document_parser
python -m src.ingestion.text_cleaner
python -m src.ingestion.legal_structure_parser
python -m src.ingestion.legal_chunker
python -m src.ingestion.index_builder
```

Lưu ý:

- không bypass login/paywall/captcha
- không crawl live trong evaluation
- ưu tiên reuse output có sẵn trong `data/processed` và `data/indexes`

## Retrieval và QA pipeline

Các module runtime chính:

- `src/retrieval/hybrid_retriever.py`
- `src/retrieval/query_router.py`
- `src/retrieval/context_expander.py`
- `src/retrieval/reranker.py`
- `src/retrieval/retrieval_pipeline.py`
- `src/generation/answer_generator.py`
- `src/qa_pipeline.py`

Luồng xử lý:

```text
question
-> hybrid retrieval
-> route selection
-> context expansion
-> reranking
-> grounded answer generation
-> citations
```

Ví dụ chạy QA:

```bash
python -m src.qa_pipeline --question "Ai khong duoc thanh lap doanh nghiep?"
python -m src.qa_pipeline --question "Khong gop du von dieu le dung han thi bi phat gi?"
python -m src.qa_pipeline --question "Nguoi nuoc ngoai gop von vao cong ty Viet Nam can dieu kien gi?"
```

Ví dụ chạy từng thành phần:

```bash
python -m src.retrieval.hybrid_retriever --query "Ai khong duoc thanh lap doanh nghiep?" --top-k 5
python -m src.retrieval.query_router --query "Khong gop du von dieu le dung han thi bi phat gi?"
python -m src.retrieval.context_expander --query "Ai khong duoc thanh lap doanh nghiep?" --top-k 5
python -m src.retrieval.reranker --query "Ai khong duoc thanh lap doanh nghiep?"
python -m src.retrieval.retrieval_pipeline --query "Ai khong duoc thanh lap doanh nghiep?"
python -m src.generation.answer_generator --query "Ai khong duoc thanh lap doanh nghiep?"
```

## Evaluation

Module:

- `src/evaluation/eval_logger.py`
- `src/evaluation/evaluate_qa.py`

Sample evaluation file:

- `data/evaluation/sample_questions.jsonl`

Chạy smoke evaluation:

```bash
python -m src.evaluation.evaluate_qa --questions data/evaluation/sample_questions.jsonl --run-id smoke_test
```

Output:

- `logs/eval_runs/<run_id>.jsonl`
- `logs/eval_runs/<run_id>_summary.json`

Metrics hiện có:

- `citation_present_rate`
- `answer_non_empty_rate`
- `route_distribution`
- `avg_context_count`
- `avg_latency_seconds`
- `legal_ref_hit_rate`

## Test

Các test chính cho ingestion + retrieval/QA:

```bash
python -m unittest ^
  tests.test_hybrid_retriever ^
  tests.test_query_router ^
  tests.test_context_expander ^
  tests.test_reranker ^
  tests.test_answer_generator ^
  tests.test_qa_pipeline ^
  tests.test_evaluation_runner
```

Test ingestion:

```bash
python -m unittest tests.test_ingestion_pipeline tests.test_ingestion_extensions
```

## Task plan

Hai task plan đã được cập nhật trạng thái hoàn thành:

- `R2AI_INGESTION_PIPELINE_TASK_PLAN.md`
- `R2AI_POST_INGESTION_RETRIEVAL_QA_TASK_PLAN.md`

## Lưu ý vận hành

- Repo hiện fallback sang offline hash embeddings nếu không load được model `intfloat/multilingual-e5-base`.
- Điều này giúp pipeline vẫn chạy được trong môi trường không có mạng, nhưng chất lượng retrieval sẽ thấp hơn model thật.
- Nếu có model local hoặc mạng ổn định, nên chuyển lại embedding runtime chuẩn để tăng chất lượng QA.
- Không bịa căn cứ pháp luật nếu context không đủ.
- Khi mở rộng tiếp, ưu tiên reuse `rag/` và `src/legal_rag/` thay vì tạo trùng logic.
