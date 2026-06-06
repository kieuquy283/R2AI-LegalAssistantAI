# R2AI Legal AI Assistant — Post-Ingestion Retrieval & QA Pipeline Task Plan

## Mục tiêu

Sau khi **Data Ingestion & Indexing Pipeline** đã hoàn thành, triển khai toàn bộ tầng:

```text
Hybrid Retriever
→ Query Router
→ Context Expansion
→ Reranker
→ Answer Generator
→ Evaluation
```

Kiến trúc mục tiêu:

```text
User Query
   ↓
Query Analyzer / Router
   ↓
Hybrid Retriever
   ↓
Seed Chunks
   ↓
Context Expansion
   ├── Parent Context
   ├── Neighbor Context
   ├── Explicit Reference / Legal Graph
   └── Cross-Domain Retrieval
   ↓
Reranker
   ↓
Answer Generator
   ↓
Evaluation / Logs
```

Hệ thống phục vụ domain:

```text
Tư vấn Luật Doanh nghiệp / SME
+ lightweight legal graph
+ controlled cross-domain retrieval
```

---

# 1. Quyền thực thi cho Codex

Codex được phép tự động thực hiện mọi command cần thiết để hoàn thành toàn bộ task.

## Cho phép

```text
- Đọc toàn bộ project trước khi sửa
- Tạo file mới
- Sửa file hiện có
- Tạo thư mục src/retrieval, src/generation, src/evaluation nếu thiếu
- Cài dependency nếu thiếu
- Chạy unit test
- Chạy validation script
- Chạy query test
- Chạy evaluation test
- Sửa lỗi rồi chạy lại đến khi pass
```

## Không cần hỏi lại confirmation

```text
- Không hỏi lại trước mỗi command
- Không hỏi lại khi cần cài dependency phổ biến
- Không hỏi lại khi cần sửa import/path/module
- Không hỏi lại khi cần tạo test hoặc validation
```

## Giới hạn

```text
- Không phá env/framework hiện tại
- Không thay embedding model/framework nếu project đã có
- Không đổi cấu trúc ingestion output nếu không cần
- Không crawl live khi trả lời/evaluation
- Không bịa căn cứ pháp luật nếu context thiếu
```

---

# 2. Quy tắc chuyển task

Chỉ được chuyển sang task tiếp theo khi task hiện tại đạt đủ:

```text
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
```

Nếu một trong bốn trạng thái chưa pass, phải sửa tiếp.

## Template trạng thái bắt buộc

Sau mỗi task, cập nhật vào file này hoặc task log:

```text
Task <TASK_NAME>
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- <file_1>
- <file_2>
Notes:
- <ghi chú nếu có>
```

---

# 3. Input bắt buộc từ ingestion

Trước khi bắt đầu, kiểm tra các file ingestion đã tồn tại:

```text
data/processed/chunks.jsonl
data/processed/context_chunks.jsonl
data/processed/legal_nodes.jsonl
data/processed/legal_edges.jsonl
data/indexes/faiss.index
data/indexes/chunk_metadata.json
```

Nếu thiếu file nào, không triển khai retrieval ngay. Phải quay lại ingestion task để build đủ.

## Validation command

```bash
python - <<'PY'
from pathlib import Path

required = [
    "data/processed/chunks.jsonl",
    "data/processed/context_chunks.jsonl",
    "data/processed/legal_nodes.jsonl",
    "data/processed/legal_edges.jsonl",
    "data/indexes/faiss.index",
    "data/indexes/chunk_metadata.json",
]

missing = [f for f in required if not Path(f).exists() or Path(f).stat().st_size == 0]
assert not missing, f"Missing ingestion outputs: {missing}"

print("INGESTION OUTPUT CHECK PASSED")
PY
```

## Status

```text
Task Check Ingestion Outputs
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- ingestion output readiness
Notes:
- Verified required ingestion outputs exist and are non-empty.
```

---

# 4. Task 1 — Hybrid Retriever

## Mục tiêu

Tạo module retrieval cơ bản để từ query lấy được seed chunks.

## Module

```text
src/retrieval/hybrid_retriever.py
```

## Input

```text
data/indexes/faiss.index
data/indexes/chunk_metadata.json
data/processed/chunks.jsonl
```

Nếu có BM25 index:

```text
data/indexes/bm25_index.pkl
```

Nếu chưa có BM25 thì dense FAISS trước, BM25 bổ sung sau.

## Output runtime

Trả về list chunk:

```json
[
  {
    "chunk_id": "...",
    "score": 0.82,
    "retrieval_score": 0.82,
    "retrieval_method": "dense",
    "content": "...",
    "embedding_text": "...",
    "metadata": {
      "doc_id": "...",
      "domain": "business_law",
      "doc_title": "...",
      "article": "Điều 17",
      "clause": "Khoản 2",
      "citation": "...",
      "source_url": "..."
    }
  }
]
```

## Yêu cầu kỹ thuật

```text
- Load FAISS index
- Load chunk_metadata.json
- Load chunks.jsonl
- Embed query bằng embedding model hiện tại của project
- Search FAISS top_k
- Map index → chunk metadata → chunk content
- Trả về chunk_id, score, content, metadata
- Có filter theo domain nếu truyền domain
```

## CLI test

```bash
python -m src.retrieval.hybrid_retriever --query "Ai không được thành lập doanh nghiệp?" --top-k 5
```

## Unit test tối thiểu

Tạo:

```text
tests/test_hybrid_retriever.py
```

Test:

```text
- load index không lỗi
- query trả về list
- mỗi result có chunk_id, score, content, metadata
- domain filter không crash
```

## Validation

```bash
python - <<'PY'
from src.retrieval.hybrid_retriever import HybridRetriever

r = HybridRetriever(
    faiss_index_path="data/indexes/faiss.index",
    metadata_path="data/indexes/chunk_metadata.json",
    chunks_path="data/processed/chunks.jsonl",
)

results = r.search("Ai không được thành lập doanh nghiệp?", top_k=5)
assert isinstance(results, list)
assert len(results) > 0, "No retrieval results"

for item in results:
    assert item.get("chunk_id")
    assert "score" in item
    assert item.get("content")
    assert item.get("metadata", {}).get("source_url")

print("Hybrid Retriever validation PASSED", {"results": len(results)})
PY
```

## Status

```text
Task Hybrid Retriever
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- src/retrieval/hybrid_retriever.py
- tests/test_hybrid_retriever.py
Notes:
- Dense + sparse retrieval works with domain filter and offline embedding fallback.
```

---

# 5. Task 2 — Query Router

## Mục tiêu

Tạo router quyết định query nên dùng route nào:

```text
SIMPLE_VECTOR
PARENT_CONTEXT
LEGAL_GRAPH_CONTEXT
CROSS_DOMAIN_CONTEXT
MULTI_DOMAIN_COMPLEX
```

## Module

```text
src/retrieval/query_router.py
```

## Input

```text
query
seed_chunks optional
```

## Output

```json
{
  "route": "CROSS_DOMAIN_CONTEXT",
  "domains": ["business_law", "administrative_penalty"],
  "needs_parent": true,
  "needs_neighbor": false,
  "needs_graph": true,
  "needs_cross_domain": true,
  "reason": "Query asks about penalty and capital contribution."
}
```

## Route logic tối thiểu

### SIMPLE_VECTOR

Dùng khi query đơn giản, không cần mở rộng.

Ví dụ:

```text
Công ty TNHH một thành viên là gì?
Doanh nghiệp tư nhân có tư cách pháp nhân không?
```

### PARENT_CONTEXT

Dùng khi query nhắc:

```text
điều
khoản
điểm
đối tượng nào
trường hợp nào
điều kiện
quyền
nghĩa vụ
```

### LEGAL_GRAPH_CONTEXT

Dùng khi query nhắc:

```text
liên quan
căn cứ
theo quy định tại
được hướng dẫn bởi
sửa đổi
bổ sung
thay thế
hết hiệu lực
còn hiệu lực
ngoại lệ
trừ trường hợp
```

### CROSS_DOMAIN_CONTEXT

Dùng khi query hoặc seed chunk chạm domain phụ:

```text
thuế
hóa đơn
lệ phí môn bài
lao động
BHXH
nhà đầu tư nước ngoài
FDI
xử phạt
mức phạt
hợp đồng
```

### MULTI_DOMAIN_COMPLEX

Dùng khi query yêu cầu tổng hợp nhiều mảng:

```text
công ty mới thành lập cần làm gì
toàn bộ thủ tục
các nghĩa vụ pháp lý
cần những văn bản nào
so sánh
```

## Domain detection

Cần detect tối thiểu:

```text
business_law
investment_law
tax_law
labor_law
social_insurance
administrative_penalty
civil_commercial_law
```

## CLI test

```bash
python -m src.retrieval.query_router --query "Không góp đủ vốn điều lệ đúng hạn thì bị phạt gì?"
```

Expected:

```text
route: CROSS_DOMAIN_CONTEXT
domains: business_law, administrative_penalty
```

## Unit test

Tạo:

```text
tests/test_query_router.py
```

Test các query mẫu:

```text
"Công ty TNHH một thành viên là gì?" → SIMPLE_VECTOR hoặc PARENT_CONTEXT
"Ai không được thành lập doanh nghiệp?" → PARENT_CONTEXT
"Không góp đủ vốn điều lệ đúng hạn thì bị phạt gì?" → CROSS_DOMAIN_CONTEXT
"Công ty mới thành lập cần làm những việc gì?" → MULTI_DOMAIN_COMPLEX
"Người nước ngoài góp vốn vào công ty Việt Nam cần điều kiện gì?" → CROSS_DOMAIN_CONTEXT với investment_law
```

## Validation

```bash
python - <<'PY'
from src.retrieval.query_router import route_query

cases = [
    ("Công ty TNHH một thành viên là gì?", {"business_law"}),
    ("Không góp đủ vốn điều lệ đúng hạn thì bị phạt gì?", {"business_law", "administrative_penalty"}),
    ("Người nước ngoài góp vốn vào công ty Việt Nam cần điều kiện gì?", {"business_law", "investment_law"}),
]

for q, expected_domains in cases:
    r = route_query(q)
    assert r["route"]
    assert expected_domains.issubset(set(r["domains"])), (q, r)

print("Query Router validation PASSED")
PY
```

## Status

```text
Task Query Router
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- src/retrieval/query_router.py
- tests/test_query_router.py
Notes:
- Route detection uses accent-insensitive keyword matching and multi-domain heuristics.
```

---

# 6. Task 3 — Context Expander

## Mục tiêu

Dùng metadata + lightweight graph để mở rộng context từ seed chunks.

## Module

```text
src/retrieval/context_expander.py
```

## Input

```text
query
route_result
seed_chunks
data/processed/chunks.jsonl
data/processed/context_chunks.jsonl
data/processed/legal_edges.jsonl
```

## Output

```json
[
  {
    "chunk_id": "...",
    "content": "...",
    "context_type": "seed",
    "relation_type": null,
    "score": 0.82,
    "metadata": {...}
  },
  {
    "chunk_id": "...",
    "content": "...",
    "context_type": "parent",
    "relation_type": "HAS_PARENT",
    "metadata": {...}
  }
]
```

## Expansion types

### Parent expansion

Nếu seed chunk có:

```text
context_chunk_id
parent_id
```

thì lấy parent/context chunk.

### Neighbor expansion

Nếu route cần neighbor, lấy:

```text
prev_chunk_id
next_chunk_id
```

Giới hạn:

```text
window = 1
không vượt doc_id
```

### Explicit reference expansion

Nếu legal_edges có:

```text
REFERS_TO
```

thì lấy target chunk/context.

Giới hạn:

```text
max_explicit_refs = 3
```

### Cross-domain expansion

Nếu route domains có domain phụ:

```text
tax_law
labor_law
investment_law
administrative_penalty
```

thì gọi retriever search thêm trong domain đó.

Giới hạn:

```text
top_k_per_satellite_domain = 3
max_context_per_satellite_domain = 2
```

## Context budget

Tối đa:

```text
simple: 5 contexts
parent: 7 contexts
graph: 9 contexts
cross-domain: 10 contexts
multi-domain: 12 contexts
```

## Priority

```text
1. seed
2. parent/context chunk
3. explicit reference
4. cross-domain direct hit
5. neighbor
```

## CLI test

```bash
python -m src.retrieval.context_expander --query "Không góp đủ vốn điều lệ đúng hạn thì bị phạt gì?" --top-k 5
```

## Unit test

Tạo:

```text
tests/test_context_expander.py
```

Test:

```text
- expand parent không crash
- expand neighbor không vượt doc_id
- expand explicit refs nếu edges có
- cross-domain expansion không vượt budget
- deduplicate chunk_id/context_chunk_id
```

## Validation

```bash
python - <<'PY'
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_router import route_query
from src.retrieval.context_expander import ContextExpander

query = "Ai không được thành lập doanh nghiệp?"

retriever = HybridRetriever(
    faiss_index_path="data/indexes/faiss.index",
    metadata_path="data/indexes/chunk_metadata.json",
    chunks_path="data/processed/chunks.jsonl",
)

seed = retriever.search(query, top_k=5)
route = route_query(query, seed_chunks=seed)

expander = ContextExpander(
    chunks_path="data/processed/chunks.jsonl",
    context_chunks_path="data/processed/context_chunks.jsonl",
    edges_path="data/processed/legal_edges.jsonl",
    retriever=retriever,
)

contexts = expander.expand(query=query, route_result=route, seed_chunks=seed)
assert contexts, "No expanded contexts"
assert len(contexts) <= 12, f"Too many contexts: {len(contexts)}"

for c in contexts:
    assert c.get("content")
    assert c.get("metadata", {}).get("source_url")

print("Context Expander validation PASSED", {"contexts": len(contexts), "route": route["route"]})
PY
```

## Status

```text
Task Context Expander
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- src/retrieval/context_expander.py
- tests/test_context_expander.py
Notes:
- Parent, neighbor, explicit reference, and cross-domain expansion respect context budgets.
```

---

# 7. Task 4 — Reranker

## Mục tiêu

Chọn context tốt nhất sau expansion.

## Module

```text
src/retrieval/reranker.py
```

## Input

```text
query
candidate_contexts
```

## Output

```json
[
  {
    "chunk_id": "...",
    "final_score": 0.91,
    "rerank_score": 0.84,
    "relation_boost": 0.1,
    "content": "...",
    "metadata": {...}
  }
]
```

## Scoring baseline

Nếu chưa có cross-encoder reranker, dùng heuristic:

```text
final_score =
    retrieval_score * 0.55
  + keyword_overlap_score * 0.25
  + relation_boost * 0.20
```

## Relation boost

```text
seed: +0.20
parent: +0.15
explicit_reference: +0.18
same_article: +0.12
cross_domain: +0.10 nếu query có domain keyword
neighbor: +0.05
```

## Yêu cầu

```text
- Deduplicate
- Không chọn quá nhiều context cùng nội dung
- Ưu tiên context có citation/source_url/article
- Có max_contexts parameter
```

## CLI test

```bash
python -m src.retrieval.reranker --query "Ai không được thành lập doanh nghiệp?"
```

## Unit test

Tạo:

```text
tests/test_reranker.py
```

Test:

```text
- rerank trả list
- final_score tồn tại
- context có source_url/citation được ưu tiên
- max_contexts hoạt động
```

## Validation

```bash
python - <<'PY'
from src.retrieval.reranker import Reranker

query = "Ai không được thành lập doanh nghiệp?"
contexts = [
    {
        "chunk_id": "a",
        "content": "Điều 17. Người không được thành lập doanh nghiệp...",
        "retrieval_score": 0.7,
        "context_type": "seed",
        "metadata": {"source_url": "x", "citation": "Luật Doanh nghiệp, Điều 17"},
    },
    {
        "chunk_id": "b",
        "content": "Tin liên quan...",
        "retrieval_score": 0.6,
        "context_type": "neighbor",
        "metadata": {},
    },
]

r = Reranker()
ranked = r.rerank(query, contexts, max_contexts=1)
assert len(ranked) == 1
assert ranked[0]["chunk_id"] == "a"
assert "final_score" in ranked[0]

print("Reranker validation PASSED")
PY
```

## Status

```text
Task Reranker
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- src/retrieval/reranker.py
- tests/test_reranker.py
Notes:
- Heuristic reranking prefers grounded contexts with citation and source metadata.
```

---

# 8. Task 5 — Retrieval Pipeline Orchestrator

## Mục tiêu

Gom các module retrieval thành pipeline end-to-end:

```text
query
→ route
→ retrieve seed chunks
→ expand context
→ rerank
→ final contexts
```

## Module

```text
src/retrieval/retrieval_pipeline.py
```

## Output

```json
{
  "query": "...",
  "route": "...",
  "domains": ["business_law"],
  "seed_chunks": [...],
  "expanded_contexts": [...],
  "final_contexts": [...]
}
```

## CLI test

```bash
python -m src.retrieval.retrieval_pipeline --query "Không góp đủ vốn điều lệ đúng hạn thì bị phạt gì?"
```

## Validation

```bash
python - <<'PY'
from src.retrieval.retrieval_pipeline import RetrievalPipeline

p = RetrievalPipeline()
result = p.run("Ai không được thành lập doanh nghiệp?")

assert result["query"]
assert result["route"]
assert result["seed_chunks"]
assert result["final_contexts"]

for c in result["final_contexts"]:
    assert c.get("content")
    assert c.get("metadata", {}).get("source_url")

print("Retrieval Pipeline validation PASSED", {
    "route": result["route"],
    "contexts": len(result["final_contexts"])
})
PY
```

## Status

```text
Task Retrieval Pipeline Orchestrator
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- src/retrieval/retrieval_pipeline.py
Notes:
- Retrieval pipeline returns route, seed chunks, expanded contexts, and final contexts.
```

---

# 9. Task 6 — Answer Generator

## Mục tiêu

Sinh câu trả lời pháp luật dựa trên final contexts.

## Module

```text
src/generation/answer_generator.py
```

## Input

```text
query
final_contexts
route_result
```

## Output

```json
{
  "answer": "...",
  "citations": [
    {
      "doc_title": "...",
      "article": "Điều 17",
      "source_url": "..."
    }
  ],
  "used_context_ids": ["..."]
}
```

## Prompt format bắt buộc

Câu trả lời phải theo cấu trúc:

```text
1. Kết luận ngắn
2. Căn cứ pháp luật
3. Phân tích áp dụng vào tình huống
4. Hướng xử lý thực tế cho SME
5. Lưu ý nếu thiếu dữ kiện
```

## Safety/legal grounding rules

```text
- Chỉ trả lời dựa trên CONTEXT được cung cấp
- Không bịa điều luật
- Không bịa số hiệu văn bản
- Bắt buộc nêu tên văn bản + điều/khoản nếu context có
- Nếu thiếu căn cứ, nói rõ chưa đủ căn cứ
- Phân biệt căn cứ chính và căn cứ liên quan
```

## Temperature

```text
Default: temperature = 0.1
Strict mode: temperature = 0.0
Practical SME mode: temperature = 0.2 max
```

## Nếu project chưa có LLM client

Tạo interface trừu tượng:

```text
generate_answer(query, contexts, model_client=None)
```

Nếu không có API key/model, dùng template-based answer fallback để test pipeline không crash.

## CLI test

```bash
python -m src.generation.answer_generator --query "Ai không được thành lập doanh nghiệp?"
```

## Unit test

Tạo:

```text
tests/test_answer_generator.py
```

Test:

```text
- output có answer
- output có citations
- answer có "Căn cứ pháp luật"
- khi context rỗng, answer nói chưa đủ căn cứ
```

## Validation

```bash
python - <<'PY'
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.generation.answer_generator import AnswerGenerator

query = "Ai không được thành lập doanh nghiệp?"

pipeline = RetrievalPipeline()
retrieval = pipeline.run(query)

gen = AnswerGenerator(temperature=0.1)
result = gen.generate(query=query, retrieval_result=retrieval)

assert result.get("answer")
assert "Căn cứ" in result["answer"] or result.get("citations")
assert result.get("used_context_ids") is not None

print("Answer Generator validation PASSED")
print(result["answer"][:500])
PY
```

## Status

```text
Task Answer Generator
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- src/generation/answer_generator.py
- tests/test_answer_generator.py
Notes:
- Generator uses grounded template fallback and never invents citations outside retrieved context.
```

---

# 10. Task 7 — End-to-End QA Pipeline

## Mục tiêu

Tạo pipeline hoàn chỉnh:

```text
question
→ retrieval pipeline
→ answer generator
→ response object
```

## Module

```text
src/qa_pipeline.py
```

## Output

```json
{
  "question": "...",
  "route": "...",
  "domains": [...],
  "answer": "...",
  "citations": [...],
  "retrieved_chunks": [...],
  "final_contexts": [...]
}
```

## CLI test

```bash
python -m src.qa_pipeline --question "Ai không được thành lập doanh nghiệp?"
```

## Validation

```bash
python - <<'PY'
from src.qa_pipeline import LegalQAPipeline

qa = LegalQAPipeline()
result = qa.answer("Ai không được thành lập doanh nghiệp?")

assert result["question"]
assert result["answer"]
assert result["route"]
assert result["final_contexts"]

print("End-to-End QA validation PASSED", {
    "route": result["route"],
    "contexts": len(result["final_contexts"])
})
print(result["answer"][:500])
PY
```

## Status

```text
Task End-to-End QA Pipeline
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- src/qa_pipeline.py
Notes:
- CLI and programmatic QA flow both pass after UTF-8 stdout handling fix.
```

---

# 11. Task 8 — Evaluation Logger

## Mục tiêu

Log đầy đủ quá trình trả lời để phục vụ evaluation.

## Module

```text
src/evaluation/eval_logger.py
```

## Log schema

```json
{
  "question_id": "...",
  "question": "...",
  "route": "...",
  "domains": ["business_law"],
  "seed_chunk_ids": ["..."],
  "expanded_context_ids": ["..."],
  "final_context_ids": ["..."],
  "citations": [...],
  "answer": "...",
  "timestamp": "..."
}
```

## Output

```text
logs/eval_runs/<run_id>.jsonl
```

## Validation

```bash
python - <<'PY'
from src.evaluation.eval_logger import EvalLogger

logger = EvalLogger(run_id="test_run")
logger.log({
    "question_id": "q1",
    "question": "test",
    "route": "SIMPLE_VECTOR",
    "domains": ["business_law"],
    "answer": "test answer",
})

path = logger.path
assert path.exists()
assert path.stat().st_size > 0

print("Evaluation Logger validation PASSED", path)
PY
```

## Status

```text
Task Evaluation Logger
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- src/evaluation/eval_logger.py
- logs/eval_runs/*.jsonl
Notes:
- Logger writes JSONL records with timestamps under logs/eval_runs.
```

---

# 12. Task 9 — Evaluation Runner

## Mục tiêu

Chạy evaluation trên tập câu hỏi test.

## Module

```text
src/evaluation/evaluate_qa.py
```

## Input

Có thể dùng:

```text
data/evaluation/questions.jsonl
```

Schema:

```json
{
  "question_id": "q1",
  "question": "Ai không được thành lập doanh nghiệp?",
  "expected_law_refs": ["Luật Doanh nghiệp 2020 Điều 17"]
}
```

Nếu chưa có file, tạo sample nhỏ:

```text
data/evaluation/sample_questions.jsonl
```

## Metrics tối thiểu

```text
- citation_present_rate
- answer_non_empty_rate
- route_distribution
- avg_context_count
- avg_latency_seconds
```

Nếu có expected refs:

```text
- legal_ref_hit_rate
```

## Output

```text
logs/eval_runs/<run_id>.jsonl
logs/eval_runs/<run_id>_summary.json
```

## CLI

```bash
python -m src.evaluation.evaluate_qa --questions data/evaluation/sample_questions.jsonl --run-id smoke_test
```

## Validation

```bash
python - <<'PY'
from pathlib import Path
import json
import subprocess
import sys

q = Path("data/evaluation/sample_questions.jsonl")
q.parent.mkdir(parents=True, exist_ok=True)

if not q.exists():
    q.write_text(
        '{"question_id":"q1","question":"Ai không được thành lập doanh nghiệp?","expected_law_refs":[]}\n',
        encoding="utf-8"
    )

subprocess.check_call([
    sys.executable,
    "-m",
    "src.evaluation.evaluate_qa",
    "--questions",
    str(q),
    "--run-id",
    "smoke_test"
])

summary = Path("logs/eval_runs/smoke_test_summary.json")
assert summary.exists()
data = json.loads(summary.read_text(encoding="utf-8"))
assert data.get("total_questions", 0) > 0
assert data.get("answer_non_empty_rate", 0) > 0

print("Evaluation Runner validation PASSED", data)
PY
```

## Status

```text
Task Evaluation Runner
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- src/evaluation/evaluate_qa.py
- data/evaluation/sample_questions.jsonl
- logs/eval_runs/*_summary.json
Notes:
- Smoke evaluation writes both run log and summary metrics JSON.
```

---

# 13. Task 10 — Final End-to-End Validation

## Mục tiêu

Đảm bảo toàn bộ pipeline sau ingestion hoạt động.

## Command tổng

```bash
python -m src.qa_pipeline --question "Ai không được thành lập doanh nghiệp?"
python -m src.qa_pipeline --question "Không góp đủ vốn điều lệ đúng hạn thì bị phạt gì?"
python -m src.qa_pipeline --question "Người nước ngoài góp vốn vào công ty Việt Nam cần điều kiện gì?"
python -m src.evaluation.evaluate_qa --questions data/evaluation/sample_questions.jsonl --run-id final_smoke_test
```

## Validation script

```bash
python - <<'PY'
from src.qa_pipeline import LegalQAPipeline

questions = [
    "Ai không được thành lập doanh nghiệp?",
    "Không góp đủ vốn điều lệ đúng hạn thì bị phạt gì?",
    "Người nước ngoài góp vốn vào công ty Việt Nam cần điều kiện gì?",
]

qa = LegalQAPipeline()

for q in questions:
    result = qa.answer(q)
    assert result.get("answer"), f"No answer for: {q}"
    assert result.get("route"), f"No route for: {q}"
    assert result.get("final_contexts"), f"No final contexts for: {q}"

    # Strongly preferred for competition scoring
    assert result.get("citations") is not None, f"No citations field for: {q}"

    print("QUESTION:", q)
    print("ROUTE:", result["route"])
    print("CONTEXTS:", len(result["final_contexts"]))
    print("ANSWER PREVIEW:", result["answer"][:300])
    print("---")

print("FINAL END-TO-END VALIDATION PASSED")
PY
```

## Status

```text
Task Final End-to-End Validation
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- full QA pipeline works
Notes:
- Final smoke test passed for three representative legal queries and evaluation run.
```

---

# 14. Final Project Status

Sau khi toàn bộ task pass, ghi:

```text
POST-INGESTION RETRIEVAL & QA PIPELINE: DONE
REVIEW: PASSED
TESTS: PASSED
VALIDATION: PASSED

Completed modules:
- src/retrieval/hybrid_retriever.py
- src/retrieval/query_router.py
- src/retrieval/context_expander.py
- src/retrieval/reranker.py
- src/retrieval/retrieval_pipeline.py
- src/generation/answer_generator.py
- src/qa_pipeline.py
- src/evaluation/eval_logger.py
- src/evaluation/evaluate_qa.py

Core capabilities:
- Dense retrieval from FAISS
- Route selection
- Parent context expansion
- Neighbor context expansion
- Lightweight legal graph expansion
- Cross-domain retrieval
- Reranking
- Citation-grounded answer generation
- Evaluation logging
```

---
