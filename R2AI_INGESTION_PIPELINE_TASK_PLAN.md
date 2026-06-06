# R2AI Legal AI Assistant — Ingestion Pipeline Task Plan

## Mục tiêu

Triển khai lần lượt các task còn lại của **Data Ingestion & Indexing Pipeline** cho kiến trúc:

```text
Hybrid RAG hiện tại
+ Hierarchical Legal Chunking
+ Lightweight Legal Graph metadata
+ FAISS Index
```

Pipeline cần hoàn thành:

```text
save Markdown
→ create manifest.jsonl
→ extract document metadata
→ clean text
→ parse Điều/Khoản
→ create chunks.jsonl
→ create context_chunks.jsonl
→ assign parent_id
→ assign prev_chunk_id / next_chunk_id
→ build FAISS index
→ save chunk_metadata.json
```

---

## 1. Quyền thực thi cho Codex

Codex được phép tự động thực hiện mọi command cần thiết để hoàn thành task.

### Cho phép

```text
- Đọc toàn bộ project trước khi sửa
- Tạo file mới
- Sửa file hiện có
- Tạo thư mục data/processed, data/indexes, data/logs nếu thiếu
- Cài package/dependency nếu thiếu
- Chạy script ingestion
- Chạy unit test
- Chạy validation script
- Chạy format/lint nếu project đang dùng
- Sửa lỗi rồi chạy lại cho đến khi pass
```

### Không cần hỏi lại confirmation

```text
- Không hỏi lại trước mỗi command
- Không hỏi lại khi cần cài dependency phổ biến
- Không hỏi lại khi cần sửa import/path/module
- Không hỏi lại khi cần tạo test hoặc validation
```

### Giới hạn

```text
- Không phá env/framework hiện tại
- Không đổi kiến trúc tổng thể nếu không cần
- Không xóa dữ liệu raw đã crawl nếu không backup
- Không bypass login/paywall/captcha
- Không crawl live trong evaluation
```

---

## 2. Quy tắc chuyển task

Codex chỉ được chuyển sang task tiếp theo khi task hiện tại đạt đủ 4 trạng thái:

```text
DONE
REVIEW PASSED
TEST PASSED
VALIDATION PASSED
```

Nếu một trong bốn trạng thái chưa pass, phải sửa tiếp.

### Template trạng thái bắt buộc

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

## 3. Definition of Done chung

Một task chỉ được coi là xong khi:

```text
[ ] Có output file đúng vị trí
[ ] Có log lỗi nếu lỗi xảy ra
[ ] Có test hoặc script validation
[ ] Test pass
[ ] Validation pass
[ ] Không làm hỏng task trước
[ ] Có thể chạy lại mà không gây lỗi nghiêm trọng
[ ] Có task log/README cập nhật trạng thái
```

---

## 4. Cấu trúc output kỳ vọng

Sau khi hoàn thành toàn bộ pipeline, project cần có:

```text
data/
├── raw/
│   ├── html/
│   ├── markdown/
│   ├── document_urls.jsonl
│   └── documents_manifest.jsonl
│
├── processed/
│   ├── documents.jsonl
│   ├── cleaned_documents.jsonl
│   ├── legal_nodes.jsonl
│   ├── chunks.jsonl
│   ├── context_chunks.jsonl
│   └── ingestion_report.json
│
├── indexes/
│   ├── faiss.index
│   └── chunk_metadata.json
│
└── logs/
    ├── crawl_documents_errors.jsonl
    └── ingestion_errors.jsonl
```

---

# TASK 1 — Save Markdown

## Mục tiêu

Đảm bảo mỗi trang văn bản chi tiết được crawl thành công đều lưu Markdown vào:

```text
data/raw/markdown/{doc_id}.md
```

## Yêu cầu kỹ thuật

```text
- Markdown lấy từ Crawl4AI result.markdown
- Ưu tiên fit_markdown nếu usable
- Nếu fit_markdown rỗng hoặc thiếu nội dung, fallback sang raw_markdown hoặc str(result.markdown)
- Không chỉ lưu HTML
- Không lưu file Markdown rỗng
- Nếu Markdown rỗng, log lỗi
```

## Field cần có trong manifest

```json
{
  "doc_id": "...",
  "source_url": "...",
  "markdown_path": "data/raw/markdown/xxx.md",
  "content_hash": "...",
  "markdown_length": 12345,
  "success": true
}
```

## Test

```bash
python -m unittest tests.test_crawl_documents

python -m src.ingestion.crawl_documents \
  --input data/raw/document_urls.jsonl \
  --html-dir data/raw/html \
  --markdown-dir data/raw/markdown \
  --manifest data/raw/documents_manifest.jsonl \
  --report data/raw/crawl_documents_report.json \
  --limit 5 \
  --rate-limit-seconds 2
```

## Validation

```text
[ ] data/raw/markdown/ tồn tại
[ ] Có ít nhất 1 file .md
[ ] File .md không rỗng
[ ] Markdown có nội dung pháp luật usable
[ ] documents_manifest.jsonl có markdown_path
[ ] Đường dẫn markdown_path tồn tại thật
```

## Status

```text
Task Save Markdown
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/raw/markdown/*.md
- data/raw/documents_manifest.jsonl
Notes:
- Reused existing crawl outputs; ran `tests.test_crawl_documents` and file-level validation on saved Markdown/manifest.
```

---

# TASK 2 — Create manifest.jsonl

## Mục tiêu

Tạo manifest cho toàn bộ văn bản đã crawl tại:

```text
data/raw/documents_manifest.jsonl
```

## Mỗi dòng manifest cần có

```json
{
  "doc_id": "...",
  "source_url": "...",
  "canonical_url": "...",
  "source_id": "...",
  "source_name": "...",
  "provider": "...",
  "domain": "business_law",
  "doc_title": "...",
  "doc_number": "...",
  "doc_type": "...",
  "issuing_body": "...",
  "issue_date": "...",
  "effective_date": "...",
  "status": "...",
  "raw_html_path": "...",
  "markdown_path": "...",
  "html_hash": "...",
  "content_hash": "...",
  "crawl_time": "...",
  "success": true
}
```

## Yêu cầu

```text
- Manifest phải là JSONL hợp lệ
- Mỗi dòng parse được bằng json.loads
- Failed crawl cũng phải ghi record success=false
- Không ghi đè manifest sai nếu chạy append
- Có thể chạy lại ở test mode
```

## Validation command

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("data/raw/documents_manifest.jsonl")
assert p.exists(), "Missing documents_manifest.jsonl"

count = 0
success = 0
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    count += 1
    if r.get("success"):
        success += 1
        assert r.get("doc_id")
        assert r.get("source_url")
        assert r.get("raw_html_path")
        assert r.get("markdown_path")
        assert Path(r["raw_html_path"]).exists()
        assert Path(r["markdown_path"]).exists()

assert count > 0, "Manifest is empty"
assert success > 0, "No successful crawled documents"
print("Manifest validation PASSED", {"records": count, "success": success})
PY
```

## Status

```text
Task Create Manifest JSONL
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/raw/documents_manifest.jsonl
Notes:
- Existing manifest parsed cleanly as JSONL and all success records pointed to real HTML/Markdown files.
```

---

# TASK 3 — Extract Document Metadata

## Mục tiêu

Trích xuất metadata cấp văn bản từ raw HTML/Markdown và tạo:

```text
data/processed/documents.jsonl
```

## Field bắt buộc

```json
{
  "doc_id": "...",
  "doc_title": "...",
  "doc_number": "...",
  "doc_type": "...",
  "issuing_body": "...",
  "signer": "...",
  "issue_date": "...",
  "effective_date": "...",
  "status": "...",
  "domain": "...",
  "source_url": "...",
  "raw_html_path": "...",
  "markdown_path": "...",
  "content_hash": "..."
}
```

## Yêu cầu

```text
- Đọc từ data/raw/documents_manifest.jsonl
- Chỉ xử lý record success=true
- Chuẩn hóa doc_id ổn định
- Không làm mất source_url
- Không bịa metadata nếu không extract được
- Nếu thiếu field thì để null, không hallucinate
```

## Module đề xuất

```text
src/ingestion/document_parser.py
```

## Command đề xuất

```bash
python -m src.ingestion.document_parser \
  --manifest data/raw/documents_manifest.jsonl \
  --output data/processed/documents.jsonl
```

## Validation

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("data/processed/documents.jsonl")
assert p.exists(), "Missing documents.jsonl"

rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
assert rows, "No documents"

required = ["doc_id", "source_url", "domain", "markdown_path", "content_hash"]
for r in rows:
    for k in required:
        assert r.get(k), f"Missing {k} in {r.get('doc_id')}"

print("Document metadata validation PASSED", {"documents": len(rows)})
PY
```

## Status

```text
Task Extract Document Metadata
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/processed/documents.jsonl
Notes:
- Added `src/ingestion/document_parser.py`; normalized boilerplate metadata to null/date instead of keeping explanatory text.
```

---

# TASK 4 — Clean Text

## Mục tiêu

Tạo bản text sạch từ Markdown để phục vụ parse cấu trúc pháp luật.

## Output

```text
data/processed/cleaned_documents.jsonl
```

## Field cần có

```json
{
  "doc_id": "...",
  "cleaned_text": "...",
  "cleaned_text_hash": "...",
  "source_url": "...",
  "domain": "..."
}
```

## Quy tắc clean

```text
- Bỏ menu/footer/breadcrumb
- Bỏ prompt đăng nhập nếu có thể
- Bỏ "tin liên quan", "xem thêm", nút chia sẻ
- Chuẩn hóa xuống dòng
- Chuẩn hóa khoảng trắng
- Giữ nguyên heading Điều/Khoản/Điểm
- Không sửa câu chữ pháp luật gốc quá mức
```

## Module đề xuất

```text
src/ingestion/text_cleaner.py
```

## Command đề xuất

```bash
python -m src.ingestion.text_cleaner \
  --documents data/processed/documents.jsonl \
  --output data/processed/cleaned_documents.jsonl
```

## Validation

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("data/processed/cleaned_documents.jsonl")
assert p.exists(), "Missing cleaned_documents.jsonl"

rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
assert rows, "No cleaned docs"

usable = 0
for r in rows:
    text = r.get("cleaned_text", "")
    assert r.get("doc_id")
    assert len(text) > 100, f"Cleaned text too short: {r.get('doc_id')}"
    if "Điều" in text or "Khoản" in text:
        usable += 1

assert usable > 0, "No legal structure signals found"
print("Clean text validation PASSED", {"documents": len(rows), "usable": usable})
PY
```

## Status

```text
Task Clean Text
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/processed/cleaned_documents.jsonl
Notes:
- Added `src/ingestion/text_cleaner.py`; trims summary/table chrome and preserves legal headings for downstream parsing.
```

---

# TASK 5 — Parse Điều/Khoản

## Mục tiêu

Parse cấu trúc pháp luật:

```text
Văn bản
→ Chương
→ Mục
→ Điều
→ Khoản
→ Điểm
```

## Module đề xuất

```text
src/ingestion/legal_structure_parser.py
```

## Output

```text
data/processed/legal_nodes.jsonl
```

## Node schema

```json
{
  "node_id": "...",
  "doc_id": "...",
  "level": "article",
  "title": "Điều 17. Quyền thành lập...",
  "article": "Điều 17",
  "article_title": "Quyền thành lập...",
  "clause": null,
  "point": null,
  "content": "...",
  "start_char": 123,
  "end_char": 456,
  "domain": "business_law",
  "source_url": "..."
}
```

## Parser rule tối thiểu

```text
- Detect Điều: ^Điều\s+\d+[a-zA-Z]?\.
- Detect Khoản: ^\d+\.
- Detect Điểm: ^[a-zđ]\)
- Detect Chương: ^Chương\s+[IVXLCDM0-9]+
- Detect Mục: ^Mục\s+\d+
```

## Command đề xuất

```bash
python -m src.ingestion.legal_structure_parser \
  --input data/processed/cleaned_documents.jsonl \
  --output data/processed/legal_nodes.jsonl
```

## Validation

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("data/processed/legal_nodes.jsonl")
assert p.exists(), "Missing legal_nodes.jsonl"

rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
assert rows, "No legal nodes"

levels = {}
for r in rows:
    levels[r["level"]] = levels.get(r["level"], 0) + 1
    assert r.get("node_id")
    assert r.get("doc_id")
    assert r.get("content") or r.get("title")

assert levels.get("article", 0) > 0, "No article nodes detected"
print("Parse Điều/Khoản validation PASSED", levels)
PY
```

## Status

```text
Task Parse Dieu Khoan
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/processed/legal_nodes.jsonl
Notes:
- Added `src/ingestion/legal_structure_parser.py`; generated chapter/section/article/clause/point nodes with parent hierarchy.
```

---

# TASK 6 — Create chunks.jsonl

## Mục tiêu

Tạo retrieval chunks nhỏ để embed.

## Output

```text
data/processed/chunks.jsonl
```

## Chunk schema

```json
{
  "chunk_id": "...",
  "doc_id": "...",
  "node_id": "...",
  "level": "clause",
  "domain": "business_law",
  "doc_title": "...",
  "article": "Điều 17",
  "clause": "Khoản 2",
  "legal_path": "Luật Doanh nghiệp 2020 > Điều 17 > Khoản 2",
  "citation": "Luật Doanh nghiệp 2020, Điều 17, Khoản 2",
  "content": "...",
  "embedding_text": "...",
  "source_url": "..."
}
```

## Rule chunking

```text
- Nếu Điều ngắn: article-level chunk
- Nếu Điều dài: split theo Khoản
- Nếu Khoản dài: split theo Điểm
- Không chunk quá nhỏ dưới 30 tokens nếu có thể merge
- Không chunk quá dài trên 1200 tokens
```

## Module đề xuất

```text
src/ingestion/legal_chunker.py
```

## Command đề xuất

```bash
python -m src.ingestion.legal_chunker \
  --nodes data/processed/legal_nodes.jsonl \
  --documents data/processed/documents.jsonl \
  --output data/processed/chunks.jsonl
```

## Validation

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("data/processed/chunks.jsonl")
assert p.exists(), "Missing chunks.jsonl"

rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
assert rows, "No chunks"

for r in rows:
    assert r.get("chunk_id")
    assert r.get("doc_id")
    assert r.get("content")
    assert r.get("embedding_text")
    assert r.get("source_url")
    assert r.get("citation")

print("chunks.jsonl validation PASSED", {"chunks": len(rows)})
PY
```

## Status

```text
Task Create Chunks JSONL
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/processed/chunks.jsonl
Notes:
- Added `src/ingestion/legal_chunker.py`; retrieval chunks favor clause/point granularity and fall back to article-level when needed.
```

---

# TASK 7 — Create context_chunks.jsonl

## Mục tiêu

Tạo context chunks lớn hơn để đưa vào LLM khi retrieval chunk nhỏ được tìm thấy.

## Output

```text
data/processed/context_chunks.jsonl
```

## Context chunk schema

```json
{
  "context_chunk_id": "...",
  "doc_id": "...",
  "level": "article",
  "domain": "business_law",
  "doc_title": "...",
  "article": "Điều 17",
  "article_title": "...",
  "legal_path": "Luật Doanh nghiệp 2020 > Điều 17",
  "citation": "Luật Doanh nghiệp 2020, Điều 17",
  "content": "...",
  "source_url": "...",
  "child_chunk_ids": ["..."]
}
```

## Rule

```text
- Context chunk ưu tiên cấp Điều
- Nếu Điều quá dài, context chunk có thể là nhóm Khoản
- Mỗi retrieval chunk phải map tới một context_chunk_id
```

## Validation

```bash
python - <<'PY'
import json
from pathlib import Path

chunks_p = Path("data/processed/chunks.jsonl")
ctx_p = Path("data/processed/context_chunks.jsonl")
assert chunks_p.exists(), "Missing chunks.jsonl"
assert ctx_p.exists(), "Missing context_chunks.jsonl"

chunks = [json.loads(x) for x in chunks_p.read_text(encoding="utf-8").splitlines() if x.strip()]
ctxs = [json.loads(x) for x in ctx_p.read_text(encoding="utf-8").splitlines() if x.strip()]

ctx_ids = {c["context_chunk_id"] for c in ctxs}
assert ctxs, "No context chunks"

missing = [c["chunk_id"] for c in chunks if c.get("context_chunk_id") and c["context_chunk_id"] not in ctx_ids]
assert not missing, f"Chunks reference missing context chunks: {missing[:5]}"

print("context_chunks.jsonl validation PASSED", {"context_chunks": len(ctxs), "chunks": len(chunks)})
PY
```

## Status

```text
Task Create Context Chunks JSONL
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/processed/context_chunks.jsonl
Notes:
- Context chunks are article-level and every retrieval chunk maps to a valid `context_chunk_id`.
```

---

# TASK 8 — Assign parent_id

## Mục tiêu

Gán quan hệ parent cho chunk/node.

## Output

Cập nhật:

```text
data/processed/chunks.jsonl
data/processed/legal_nodes.jsonl
```

Hoặc tạo thêm:

```text
data/processed/legal_edges.jsonl
```

## Edge schema

```json
{
  "source_id": "child_id",
  "target_id": "parent_id",
  "relation_type": "HAS_PARENT",
  "confidence": 1.0
}
```

## Validation

```bash
python - <<'PY'
import json
from pathlib import Path

chunks = [json.loads(x) for x in Path("data/processed/chunks.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
chunk_ids = {c["chunk_id"] for c in chunks}
node_ids = set()
nodes_p = Path("data/processed/legal_nodes.jsonl")
if nodes_p.exists():
    node_ids = {json.loads(x)["node_id"] for x in nodes_p.read_text(encoding="utf-8").splitlines() if x.strip()}

has_parent = 0
for c in chunks:
    pid = c.get("parent_id")
    if pid:
        has_parent += 1
        assert pid in chunk_ids or pid in node_ids, f"parent_id not found: {pid}"

assert has_parent > 0, "No parent_id assigned"
print("parent_id validation PASSED", {"chunks_with_parent": has_parent})
PY
```

## Status

```text
Task Assign Parent ID
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/processed/chunks.jsonl
- data/processed/legal_edges.jsonl
Notes:
- Parent links are stored in chunk/node records and mirrored into `data/processed/legal_edges.jsonl`.
```

---

# TASK 9 — Assign prev_chunk_id / next_chunk_id

## Mục tiêu

Gán quan hệ neighborhood để phục vụ neighbor expansion.

## Output

Cập nhật:

```text
data/processed/chunks.jsonl
```

Và/hoặc:

```text
data/processed/legal_edges.jsonl
```

## Field

```json
{
  "prev_chunk_id": "...",
  "next_chunk_id": "..."
}
```

## Rule

```text
- Gán prev/next theo thứ tự xuất hiện trong cùng doc_id
- Ưu tiên cùng level nếu có thể
- Không nối chunk khác document
```

## Validation

```bash
python - <<'PY'
import json
from pathlib import Path

chunks = [json.loads(x) for x in Path("data/processed/chunks.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
by_id = {c["chunk_id"]: c for c in chunks}

links = 0
for c in chunks:
    cid = c["chunk_id"]
    for key in ["prev_chunk_id", "next_chunk_id"]:
        other = c.get(key)
        if other:
            links += 1
            assert other in by_id, f"{key} not found: {other}"
            assert by_id[other]["doc_id"] == c["doc_id"], f"{key} crosses document boundary: {cid} -> {other}"

assert links > 0, "No prev/next links assigned"
print("prev/next validation PASSED", {"links": links})
PY
```

## Status

```text
Task Assign Prev Next Chunk IDs
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/processed/chunks.jsonl
- data/processed/legal_edges.jsonl
Notes:
- Neighbor links are assigned by in-document appearance order using `start_char`.
```

---

# TASK 10 — Build FAISS Index

## Mục tiêu

Build dense vector index từ `embedding_text` trong:

```text
data/processed/chunks.jsonl
```

## Output

```text
data/indexes/faiss.index
```

## Yêu cầu

```text
- Dùng embedding model hiện có trong project nếu đã có
- Không thay đổi framework/model nếu không cần
- Nếu project đã có index builder cũ, reuse và mở rộng
- Nếu chưa có, tạo src/ingestion/index_builder.py
- Index order phải khớp với chunk_metadata.json
```

## Command đề xuất

```bash
python -m src.ingestion.index_builder \
  --chunks data/processed/chunks.jsonl \
  --faiss-index data/indexes/faiss.index \
  --metadata data/indexes/chunk_metadata.json
```

## Dependency

Nếu thiếu:

```bash
pip install faiss-cpu sentence-transformers
```

Nhưng ưu tiên dùng dependency/model đã có trong project.

## Validation

```bash
python - <<'PY'
from pathlib import Path

p = Path("data/indexes/faiss.index")
assert p.exists(), "Missing faiss.index"
assert p.stat().st_size > 0, "faiss.index is empty"

print("FAISS index validation PASSED", {"size_bytes": p.stat().st_size})
PY
```

## Status

```text
Task Build FAISS Index
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/indexes/faiss.index
Notes:
- Added `src/ingestion/index_builder.py`; reused project embedding interface and added offline hash-embedding fallback when the configured HF model is unavailable.
```

---

# TASK 11 — Save chunk_metadata.json

## Mục tiêu

Lưu metadata mapping theo đúng thứ tự vector trong FAISS.

## Output

```text
data/indexes/chunk_metadata.json
```

## Schema

```json
[
  {
    "index": 0,
    "chunk_id": "...",
    "doc_id": "...",
    "domain": "business_law",
    "doc_title": "...",
    "article": "Điều 17",
    "clause": "Khoản 2",
    "citation": "...",
    "source_url": "...",
    "context_chunk_id": "...",
    "parent_id": "...",
    "prev_chunk_id": "...",
    "next_chunk_id": "..."
  }
]
```

## Validation

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("data/indexes/chunk_metadata.json")
assert p.exists(), "Missing chunk_metadata.json"

rows = json.loads(p.read_text(encoding="utf-8"))
assert isinstance(rows, list), "chunk_metadata must be a list"
assert rows, "chunk_metadata is empty"

for i, r in enumerate(rows):
    assert r.get("index") == i, f"Index mismatch at {i}"
    assert r.get("chunk_id")
    assert r.get("doc_id")
    assert r.get("source_url")
    assert r.get("citation")

print("chunk_metadata validation PASSED", {"records": len(rows)})
PY
```

## Status

```text
Task Save Chunk Metadata JSON
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/indexes/chunk_metadata.json
Notes:
- Metadata order matches FAISS vector insertion order exactly.
```

---

# FINAL — End-to-End Validation

Sau khi tất cả task trên pass, chạy validation tổng.

## Command

```bash
python - <<'PY'
import json
from pathlib import Path

required_files = [
    "data/raw/documents_manifest.jsonl",
    "data/processed/documents.jsonl",
    "data/processed/cleaned_documents.jsonl",
    "data/processed/legal_nodes.jsonl",
    "data/processed/chunks.jsonl",
    "data/processed/context_chunks.jsonl",
    "data/indexes/faiss.index",
    "data/indexes/chunk_metadata.json",
]

for f in required_files:
    p = Path(f)
    assert p.exists(), f"Missing {f}"
    assert p.stat().st_size > 0, f"Empty {f}"

chunks = [json.loads(x) for x in Path("data/processed/chunks.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
metadata = json.loads(Path("data/indexes/chunk_metadata.json").read_text(encoding="utf-8"))

assert len(chunks) == len(metadata), f"chunks != metadata: {len(chunks)} != {len(metadata)}"

required_chunk_fields = [
    "chunk_id",
    "doc_id",
    "domain",
    "content",
    "embedding_text",
    "source_url",
    "citation",
]

for c in chunks:
    for field in required_chunk_fields:
        assert c.get(field), f"Missing {field} in chunk {c.get('chunk_id')}"

print("END-TO-END INGESTION VALIDATION PASSED")
print({
    "chunks": len(chunks),
    "metadata": len(metadata),
})
PY
```

## Final Status

```text
Pipeline Final Validation
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- all required ingestion/index files
Notes:
- End-to-end validation passed with 3117 chunks and matching metadata rows.
```

INGESTION PIPELINE: DONE
REVIEW: PASSED
TESTS: PASSED
VALIDATION: PASSED

---
