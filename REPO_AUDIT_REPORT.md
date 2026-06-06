# Repo Audit Report

Audit scope: current repository state against the target R2AI Legal Assistant architecture for ingestion, chunking, indexing, lightweight legal graph, retrieval, QA, and evaluation.

Status legend:

- `EXISTS`: implementation/data is present and usable
- `PARTIAL`: implementation exists but has architectural, data-quality, or integration gaps
- `MISSING`: expected capability or entrypoint is not present

## Summary

- The repo already contains a full offline ingestion path under `src/ingestion/`.
- The repo already contains a separate retrieval/QA path under `src/retrieval/`, `src/generation/`, `src/evaluation/`, and `src/qa_pipeline.py`.
- Core processed artifacts exist and are non-empty: `documents.jsonl`, `cleaned_documents.jsonl`, `legal_nodes.jsonl`, `chunks.jsonl`, `context_chunks.jsonl`, `legal_edges.jsonl`, `faiss.index`, `chunk_metadata.json`, `bm25_corpus.json`, `bm25_metadata.json`.
- The main gaps are not raw absence, but integration and quality:
  - metadata extraction quality is weak
  - configs/scripts still point partly to legacy corpus/index paths
  - legal graph is split across multiple files and `legal_edges.jsonl` does not contain explicit-reference or cross-domain relations
  - router / expander / reranker / generator are functional but still heuristic-heavy
  - there is no single canonical end-to-end ingestion orchestrator script

## Audit Table

| Area | Requirement | Status | Current file/module | Gap | Recommended action | Priority |
|---|---|---|---|---|---|---|
| Ingestion Sources | Source registry YAML with domains, crawl policy, compliance rules | EXISTS | `data/sources/sources.yaml`, `src/ingestion/source_registry.py` | File exists and validator exists, but YAML content shows mojibake/encoding corruption in many Vietnamese labels | Normalize `sources.yaml` encoding to UTF-8 clean text and add schema test for human-readable fields | Medium |
| Ingestion Sources | Domain taxonomy for routing/cross-domain logic | EXISTS | `data/sources/domain_taxonomy.json` | Present and used by router/reference enrichment | Keep taxonomy as single source of truth and document keyword maintenance process | Low |
| URL Collection | Collect detail links from configured public search pages | EXISTS | `src/ingestion/collect_urls.py` | Functional CLI exists, but depends on `crawl4ai` runtime and conservative pagination only | Add provider-specific pagination/link extraction adapters if coverage needs to expand | Medium |
| Detail Crawl | Crawl public detail pages, save HTML and Markdown | EXISTS | `src/ingestion/crawl_documents.py` | Functional and writes manifest/html/markdown; access restriction detection is heuristic only | Add provider-specific extraction sanity checks and stronger restriction classification | Medium |
| Manifest | Persist manifest of crawled documents with metadata and file paths | EXISTS | `data/raw/documents_manifest.jsonl`, `src/ingestion/crawl_documents.py` | Manifest exists and has required fields | Keep as canonical ingestion input for downstream steps | Low |
| Metadata Extraction | Extract structured document metadata from manifest/detail pages | PARTIAL | `src/ingestion/document_parser.py`, `data/processed/documents.jsonl` | Module exists, but current outputs show suspicious values like wrong `issue_date`/`effective_date` inherited from boilerplate | Replace generic label scraping with provider-aware metadata extraction and validation rules | High |
| Text Cleaning | Clean markdown into parser-friendly legal text | EXISTS | `src/ingestion/text_cleaner.py`, `data/processed/cleaned_documents.jsonl` | Working cleaner exists; no obvious blocking gap from artifact inspection | Add regression tests against known noisy patterns from current providers | Medium |
| Điều/Khoản Parsing | Parse chapter/section/article/clause/point hierarchy | EXISTS | `src/ingestion/legal_structure_parser.py`, `data/processed/legal_nodes.jsonl` | Nodes are produced at scale; regex parser is still format-sensitive | Add parser fixtures for alternate formats and mixed markup edge cases | Medium |
| Chunks | Create retrieval chunks with citation, legal path, embedding text | EXISTS | `src/ingestion/legal_chunker.py`, `data/processed/chunks.jsonl` | Artifact exists with 3117 rows and expected fields | Keep current shape stable; add schema contract tests on chunk fields | Low |
| Context Chunks | Create parent/article-level context chunks | EXISTS | `src/ingestion/legal_chunker.py`, `data/processed/context_chunks.jsonl` | Artifact exists with 446 rows | Add explicit schema docs for context chunk semantics | Low |
| Parent Links | Assign `parent_id` from legal hierarchy | PARTIAL | `src/ingestion/legal_chunker.py`, `data/processed/chunks.jsonl`, `data/processed/sanity_report.json` | Most chunks have `parent_id`, but sanity report still lists some chunks without parent | Define acceptable root-node cases vs true parent-link defects, then enforce validation rule | Medium |
| Prev/Next Links | Assign `prev_chunk_id` / `next_chunk_id` within document | EXISTS | `src/ingestion/legal_chunker.py`, `data/processed/chunks.jsonl` | Coverage is high and sanity report shows no dangling neighbors | Keep current invariant check in sanity report | Low |
| Legal Graph | Build lightweight legal graph including parent/neighbor/explicit refs/cross-domain | PARTIAL | `data/processed/legal_edges.jsonl`, `data/processed/explicit_refs.jsonl`, `data/processed/cross_domain_edges.jsonl`, `src/ingestion/reference_enricher.py` | Graph is split across 3 files; `legal_edges.jsonl` only contains `HAS_PARENT`, `PREV_CHUNK`, `NEXT_CHUNK`; explicit refs and cross-domain relations are not merged into one graph artifact | Decide on a canonical graph format and either merge all relations into `legal_edges.jsonl` or document multi-file graph contract clearly | High |
| Explicit References | Resolve article-to-article references | PARTIAL | `src/ingestion/reference_enricher.py`, `data/processed/explicit_refs.jsonl` | Feature exists and 242 refs resolve, but many unresolved refs remain in sanity report | Add unresolved-ref diagnostics by doc/provider and improve article lookup / aliasing | High |
| Cross-Domain Edges | Infer cross-domain relations from taxonomy keywords | EXISTS | `src/ingestion/reference_enricher.py`, `data/processed/cross_domain_edges.jsonl` | Artifact exists and is consumed by context expander | Tighten scoring/thresholds and add explainability fields if needed | Medium |
| FAISS Index | Build dense vector index from chunks | EXISTS | `src/ingestion/index_builder.py`, `data/indexes/faiss.index` | Working index exists and CLI exists | Keep current output contract; add checksum/version metadata if reproducibility matters | Low |
| Chunk Metadata | Persist retrieval metadata aligned with FAISS rows | EXISTS | `src/ingestion/index_builder.py`, `data/indexes/chunk_metadata.json` | Artifact exists and includes parent/context/prev/next fields | Keep as primary mapping file for dense retrieval | Low |
| BM25 Indexing | Build sparse retrieval corpus/metadata artifacts | EXISTS | `src/ingestion/bm25_builder.py`, `data/indexes/bm25_corpus.json`, `data/indexes/bm25_metadata.json` | Sparse artifacts exist; no pickled BM25 index, runtime rebuild happens in retriever | If startup time matters, persist a prebuilt sparse index or cache token statistics | Medium |
| Incremental Update | Incrementally rebuild processed artifacts from manifest deltas | EXISTS | `src/ingestion/incremental_update.py`, `data/processed/incremental_state.json` | Functional module exists | Add end-to-end regression around add/change/remove manifest cases | Medium |
| Sanity Report | Generate ingestion health report | EXISTS | `src/ingestion/sanity_report.py`, `data/processed/sanity_report.json` | Report exists and already catches missing context/dangling neighbor/noise/unresolved refs | Extend report to include metadata quality anomalies and domain coverage warnings | Medium |
| Hybrid Retriever | Dense + BM25 retrieval over chunk artifacts | EXISTS | `src/retrieval/hybrid_retriever.py` | Functional retriever exists with domain filter and fallback BM25 implementation | Add retrieval quality benchmarks and expose candidate pool tuning via config | Medium |
| Query Router | Route query among simple/parent/graph/cross-domain/multi-domain modes | EXISTS | `src/retrieval/query_router.py` | Functional router exists; keyword heuristics only | Externalize routing rules to config or add learned/classifier-based routing later | Medium |
| Context Expansion | Expand seed results with parent/explicit refs/cross-domain/neighbor | PARTIAL | `src/retrieval/context_expander.py` | Parent, graph, cross-domain, neighbor logic exists, but graph expansion depends on `explicit_refs` embedded back into `chunks.jsonl` rather than consuming a unified graph source | Refactor expander to consume canonical graph artifacts directly and document precedence/budgets | High |
| Reranker | Re-rank expanded contexts | PARTIAL | `src/retrieval/reranker.py` | Heuristic reranker exists; no cross-encoder, no semantic dedup beyond `chunk_id`, no explicit same-article feature despite intended scoring baseline | Add stronger dedup/grouping and optionally plug in cross-encoder behind feature flag | Medium |
| Retrieval Orchestration | End-to-end retrieval pipeline | EXISTS | `src/retrieval/retrieval_pipeline.py` | Functional orchestrator exists | Keep as canonical retrieval runtime entrypoint | Low |
| Answer Generator | Grounded legal answer generation with citations and missing-context fallback | PARTIAL | `src/generation/answer_generator.py` | Functional template-based answerer exists, but no real LLM integration path in this module and answer quality depends heavily on first retrieved chunk | Add optional model client interface and grounding checks before selecting lead basis | High |
| QA Pipeline | End-to-end question -> retrieval -> answer response object | EXISTS | `src/qa_pipeline.py` | Working CLI/module exists | Keep as primary QA runtime entrypoint | Low |
| Evaluation Logger | Log question/route/context/citation/answer traces | EXISTS | `src/evaluation/eval_logger.py` | Present and simple | Add schema version and optional run metadata header if experiments increase | Low |
| Evaluation Runner | Run batch evaluation and emit summary metrics | EXISTS | `src/evaluation/evaluate_qa.py` | Functional runner exists and writes JSONL + summary | Add retrieval-grounding metrics and hallucination checks if legal QA scoring expands | Medium |
| Dependencies | Python dependencies for crawl, indexing, retrieval, QA | PARTIAL | `requirements.txt` | Core packages exist, but environment setup for `crawl4ai`/Playwright and local embedding model availability is implicit, not enforced; repo also depends on offline fallback behavior | Document bootstrap steps explicitly and separate optional/runtime-heavy dependencies from core | Medium |
| Data Files | Required raw/processed/index artifacts present | PARTIAL | `data/raw/*`, `data/processed/*`, `data/indexes/*` | Core files exist, but current built dataset only contains 5 crawled docs and all processed docs are `business_law`; satellite-domain coverage is weak relative to target architecture | Expand source coverage and rebuild processed artifacts for true multi-domain retrieval validation | High |
| Scripts | Operational scripts for build/eval/submission | PARTIAL | `scripts/`, `src/*` CLIs | Repo contains both legacy `scripts/*` pipeline and newer `src/*` pipeline; operational surface is split and partially inconsistent | Mark canonical scripts vs legacy scripts, or move legacy-only paths under `legacy/`/`docs` | High |
| Entrypoints | Clear end-to-end entrypoints for ingestion, retrieval, QA, evaluation | PARTIAL | `python -m src.ingestion.*`, `python -m src.qa_pipeline`, `python -m src.evaluation.evaluate_qa`, legacy `scripts/*` | Component CLIs exist, but there is no single authoritative end-to-end ingestion runner; configs still reference older paths like `data/indexes/default` and `data/processed/articles.jsonl` | Add one canonical orchestrator command and align config paths with current `src/ingestion` outputs | High |
| Config Alignment | Retrieval/generation/evaluation configs aligned with current pipeline | PARTIAL | `configs/retrieval.yaml`, `configs/generation.yaml`, `configs/evaluation.yaml` | Configs still point partly to legacy corpus/index paths and not to `data/indexes/faiss.index` / `data/processed/chunks.jsonl` runtime used by new pipeline | Either retire legacy configs or create new configs for the current ingestion/retrieval stack | High |

## Key Findings

### 1. The repo already has the target architecture in code form

The major building blocks are present:

- ingestion: `src/ingestion/*`
- retrieval: `src/retrieval/*`
- generation: `src/generation/*`
- QA pipeline: `src/qa_pipeline.py`
- evaluation: `src/evaluation/*`

This is not a greenfield repo anymore. The main issue is consolidation and quality, not absence.

### 2. The repository currently carries two parallel worlds

There is a newer ingestion/retrieval/QA stack under `src/`, but there is also a legacy/older operational stack under:

- `scripts/*`
- `rag/*`
- `src/legal_rag/*`

This creates ambiguity about:

- which index is canonical
- which corpus is canonical
- which evaluation path is canonical
- which config files are authoritative

### 3. The lightweight legal graph is present but fragmented

Current graph-related artifacts are split into:

- `data/processed/legal_edges.jsonl`
- `data/processed/explicit_refs.jsonl`
- `data/processed/cross_domain_edges.jsonl`

This means the graph exists conceptually, but there is no single canonical graph contract yet.

### 4. Metadata quality is the most visible ingestion weakness

From sampled outputs:

- `documents_manifest.jsonl` contains boilerplate values for `issue_date`, `effective_date`, and `status`
- `documents.jsonl` shows suspicious normalized metadata values

This is the highest-value ingestion quality gap because it affects citations, legal filtering, and answer grounding.

### 5. Current data coverage is structurally valid but domain-thin

Observed built artifacts:

- `5` crawled documents
- all processed docs currently in `business_law`

The architecture supports multi-domain behavior, but the current built corpus does not yet reflect strong satellite-domain coverage.

## Suggested Priority Order

1. Align canonical runtime/config/script surface
2. Fix metadata extraction quality
3. Unify legal graph contract
4. Expand multi-domain data coverage
5. Strengthen context expansion/reranking/answer grounding quality

## Referenced Modules and Artifacts

- Source registry: `data/sources/sources.yaml`, `src/ingestion/source_registry.py`
- Crawl pipeline: `src/ingestion/collect_urls.py`, `src/ingestion/crawl_documents.py`
- Processing: `src/ingestion/document_parser.py`, `src/ingestion/text_cleaner.py`, `src/ingestion/legal_structure_parser.py`, `src/ingestion/legal_chunker.py`, `src/ingestion/reference_enricher.py`
- Indexing: `src/ingestion/bm25_builder.py`, `src/ingestion/index_builder.py`
- Validation: `src/ingestion/incremental_update.py`, `src/ingestion/sanity_report.py`
- Retrieval: `src/retrieval/hybrid_retriever.py`, `src/retrieval/query_router.py`, `src/retrieval/context_expander.py`, `src/retrieval/reranker.py`, `src/retrieval/retrieval_pipeline.py`
- Generation: `src/generation/answer_generator.py`
- QA: `src/qa_pipeline.py`
- Evaluation: `src/evaluation/eval_logger.py`, `src/evaluation/evaluate_qa.py`
