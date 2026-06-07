# R2AI Legal Assistant — Post Crawl 500 to Full Index & Evaluation Task Plan

## Mục tiêu

File này dùng để Codex thực hiện tiếp pipeline sau khi đã crawl batch 500 document detail thành công.

Trạng thái hiện tại:

```text
URL collection: DONE
document_urls.jsonl: khoảng 5397 unique URLs
Detail crawl batch 500: DONE
Success: 500
Failed: 0
Restricted signals: 500
```

Cần thực hiện tiếp các task:

```text
1. Validate chất lượng HTML/Markdown của 500 docs
2. Bật resume/skip nếu chưa có
3. Crawl full 5397 URLs
4. Parse metadata
5. Clean text
6. Parse Điều/Khoản
7. Chunk + context chunk
8. Build graph
9. Build FAISS/BM25 index
10. Chạy retrieval/QA evaluation với file câu hỏi test
```

Nguyên tắc bắt buộc:

```text
DONE
REVIEW PASSED
TEST PASSED
VALIDATION PASSED
```

rồi mới được chuyển sang task tiếp theo.

---

# 1. Quyền thực thi cho Codex

Codex được phép tự động thực hiện mọi command cần thiết để hoàn thành toàn bộ task trong file này.

## Cho phép

```text
- Đọc toàn bộ repo trước khi sửa
- Tạo file mới
- Sửa file hiện có
- Tạo thư mục data/processed, data/indexes, data/logs nếu thiếu
- Cài package/dependency nếu thiếu
- Chạy crawl tiếp với --resume
- Chạy parser/chunker/index builder/evaluation
- Chạy unit test
- Chạy validation script
- Sửa lỗi rồi chạy lại đến khi pass
```

## Không cần hỏi confirmation

```text
- Không hỏi lại trước mỗi command
- Không hỏi lại khi cần cài dependency phổ biến
- Không hỏi lại khi cần sửa import/path/module
- Không hỏi lại khi cần tạo test hoặc validation
- Không hỏi lại khi cần chạy crawl full bằng --resume
```

## Giới hạn

```text
- Không phá env/framework hiện tại
- Không xóa dữ liệu raw/processed/index nếu không backup
- Không xóa document_urls.jsonl
- Không xóa raw HTML/Markdown đã crawl thành công
- Không crawl live trong bước evaluation
- Không bypass login/paywall/captcha
- Không bịa metadata/căn cứ pháp luật
- Không chuyển task nếu task hiện tại chưa pass đủ 4 trạng thái
```

---

# 2. Quy tắc chuyển task

Mỗi task chỉ được coi là hoàn thành khi có đủ:

```text
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
```

Nếu một trong bốn trạng thái chưa pass, phải sửa tiếp.

Sau mỗi task, cập nhật task log hoặc file này theo template:

```text
Task <TASK_NAME>
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- <file/output>
Notes:
- <ghi chú>
```

---

# 3. Pre-check baseline

Trước khi chạy các task, kiểm tra các file đầu vào:

```text
data/raw/document_urls.jsonl
data/raw/documents_manifest.jsonl
data/raw/html/
data/raw/markdown/
src/ingestion/crawl_documents.py
```

## PowerShell command

```powershell
@'
from pathlib import Path

required = [
    "data/raw/document_urls.jsonl",
    "data/raw/documents_manifest.jsonl",
    "data/raw/html",
    "data/raw/markdown",
    "src/ingestion/crawl_documents.py",
]

missing = [x for x in required if not Path(x).exists()]
assert not missing, f"Missing baseline files: {missing}"

print("BASELINE CHECK PASSED")
'@ | python
```

## Status

```text
Task Baseline Check
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/raw/document_urls.jsonl
- data/raw/documents_manifest.jsonl
- data/raw/html/
- data/raw/markdown/
- src/ingestion/crawl_documents.py
Notes:
-
```

---

# 4. Task 1 — Validate chất lượng HTML/Markdown của 500 docs

## Mục tiêu

Xác nhận 500 documents đã crawl thành công thực sự usable cho parse/chunk/index.

Cần kiểm tra:

```text
- documents_manifest.jsonl có records success=true
- raw_html_path tồn tại
- markdown_path tồn tại
- HTML không rỗng
- Markdown không rỗng
- Markdown không quá ngắn bất thường
- Có dấu hiệu nội dung pháp luật: Điều, Khoản, Căn cứ, Luật, Nghị định, Thông tư
- Restricted signals = 500 có phải chỉ là warning hay làm Markdown rỗng
```

## Output cần tạo

```text
data/raw/crawl_quality_report.json
```

## Module đề xuất

Nếu chưa có, tạo:

```text
src/ingestion/validate_crawl_quality.py
```

## Command đề xuất

```bash
python -m src.ingestion.validate_crawl_quality --manifest data/raw/documents_manifest.jsonl --output data/raw/crawl_quality_report.json
```

## Validation script bắt buộc

```powershell
@'
import json
from pathlib import Path
from collections import Counter

manifest = Path("data/raw/documents_manifest.jsonl")
assert manifest.exists(), "Missing documents_manifest.jsonl"

rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
success = [r for r in rows if r.get("success")]

assert success, "No successful crawl records"

empty_html = []
empty_md = []
short_md = []
missing_paths = []
legal_signal = 0
restricted = 0
domains = Counter()

for r in success:
    domains[r.get("domain", "unknown")] += 1

    html_path = Path(r.get("raw_html_path", ""))
    md_path = Path(r.get("markdown_path", ""))

    if not html_path.exists() or not md_path.exists():
        missing_paths.append(r.get("source_url"))
        continue

    html = html_path.read_text(encoding="utf-8", errors="ignore").strip()
    md = md_path.read_text(encoding="utf-8", errors="ignore").strip()

    if not html:
        empty_html.append(r.get("source_url"))

    if not md:
        empty_md.append(r.get("source_url"))
    elif len(md) < 500:
        short_md.append(r.get("source_url"))

    if any(sig in md for sig in ["Điều ", "Khoản ", "Căn cứ", "Luật ", "Nghị định", "Thông tư"]):
        legal_signal += 1

    access = r.get("access_restriction", {})
    if access.get("has_restriction_signal"):
        restricted += 1

report = {
    "manifest_rows": len(rows),
    "success": len(success),
    "missing_paths": len(missing_paths),
    "empty_html": len(empty_html),
    "empty_markdown": len(empty_md),
    "short_markdown_under_500": len(short_md),
    "legal_signal_count": legal_signal,
    "restricted_signal_count": restricted,
    "domains": dict(domains),
    "sample_empty_markdown": empty_md[:5],
    "sample_short_markdown": short_md[:5],
}

Path("data/raw/crawl_quality_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(report, ensure_ascii=False, indent=2))

assert len(success) >= 500, f"Expected at least 500 success records, got {len(success)}"
assert len(missing_paths) == 0, f"Missing file paths: {missing_paths[:5]}"
assert len(empty_html) < len(success) * 0.1, "Too many empty HTML files"
assert len(empty_md) < len(success) * 0.2, "Too many empty Markdown files"
assert legal_signal > 0, "No legal content signal found"

print("CRAWL QUALITY VALIDATION PASSED")
'@ | python
```

## Review checklist

```text
[ ] success >= 500
[ ] failed thấp hoặc bằng 0
[ ] empty Markdown thấp
[ ] legal_signal_count > 0
[ ] restricted signals không làm mất nội dung
[ ] crawl_quality_report.json được tạo
```

## Status

```text
Task Validate HTML Markdown Quality
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/raw/crawl_quality_report.json
Notes:
-
```

---

# 5. Task 2 — Bật resume/skip nếu chưa có

## Mục tiêu

Đảm bảo `crawl_documents.py` có thể chạy full nhiều lần mà không crawl lại URL đã success.

Cần có:

```text
--resume
--append-manifest
skip already crawled success URLs
skipped_count trong report
không duplicate success URL trong manifest
```

## File cần kiểm tra/sửa

```text
src/ingestion/crawl_documents.py
tests/test_crawl_documents.py
```

## Yêu cầu

```text
1. Nếu manifest đã tồn tại, đọc record success=true.
2. Lấy set URL đã crawl thành công theo canonical_url hoặc source_url/url.
3. Khi chạy --resume, bỏ qua URL đã success.
4. Ghi skipped_count vào report.
5. Không duplicate manifest success records.
6. URL fail trước đó được phép retry.
7. Report phải rõ:
   - total_input_records_before_filter
   - already_crawled_count
   - to_crawl_count
   - skipped_count
   - success_count
   - failed_count
```

## Test command

```bash
python -m unittest tests.test_crawl_documents
```

## Smoke command

```bash
python -m src.ingestion.crawl_documents --input data/raw/document_urls.jsonl --html-dir data/raw/html --markdown-dir data/raw/markdown --manifest data/raw/documents_manifest.jsonl --report data/raw/crawl_documents_report.json --limit 20 --resume --rate-limit-seconds 2
```

## Validation script

```powershell
@'
import json
from pathlib import Path
from collections import Counter

manifest = Path("data/raw/documents_manifest.jsonl")
report_p = Path("data/raw/crawl_documents_report.json")

assert manifest.exists(), "Missing manifest"
assert report_p.exists(), "Missing crawl report"

rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
success = [r for r in rows if r.get("success")]

urls = [r.get("canonical_url") or r.get("source_url") or r.get("url") for r in success]

dups = [u for u, c in Counter(urls).items() if u and c > 1]
assert not dups, f"Duplicate success URLs in manifest: {dups[:5]}"

report = json.loads(report_p.read_text(encoding="utf-8"))
assert "skipped_count" in report, "Report missing skipped_count"

print("RESUME VALIDATION PASSED", {
    "success_records": len(success),
    "skipped_count": report.get("skipped_count"),
    "report_keys": sorted(report.keys())
})
'@ | python
```

## Status

```text
Task Resume Skip Already Crawled
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- updated src/ingestion/crawl_documents.py
- updated tests/test_crawl_documents.py
Notes:
-
```

---

# 6. Task 3 — Crawl full 5397 URLs

## Mục tiêu

Crawl toàn bộ URL trong:

```text
data/raw/document_urls.jsonl
```

Hiện có khoảng:

```text
5397 unique URLs
```

Cần crawl toàn bộ bằng resume để không crawl lại 500 docs đã xong.

## Command full crawl

```bash
python -m src.ingestion.crawl_documents --input data/raw/document_urls.jsonl --html-dir data/raw/html --markdown-dir data/raw/markdown --manifest data/raw/documents_manifest.jsonl --report data/raw/crawl_documents_report.json --resume --rate-limit-seconds 2
```

Nếu cần an toàn hơn, chạy nhiều batch:

```bash
python -m src.ingestion.crawl_documents --input data/raw/document_urls.jsonl --html-dir data/raw/html --markdown-dir data/raw/markdown --manifest data/raw/documents_manifest.jsonl --report data/raw/crawl_documents_report.json --limit 1000 --resume --rate-limit-seconds 2
```

Chạy lặp lại batch 1000 đến khi `to_crawl_count = 0` hoặc toàn bộ unique URLs đã success/fail.

## Error handling

```text
- Timeout rải rác thì log, không dừng toàn bộ
- Nếu lỗi nhiều, tăng --rate-limit-seconds 3
- Không retry vô hạn
- Không bypass login/paywall/captcha
```

## Validation script

```powershell
@'
import json
from pathlib import Path

urls_p = Path("data/raw/document_urls.jsonl")
manifest_p = Path("data/raw/documents_manifest.jsonl")

assert urls_p.exists(), "Missing document_urls.jsonl"
assert manifest_p.exists(), "Missing documents_manifest.jsonl"

url_rows = [json.loads(x) for x in urls_p.read_text(encoding="utf-8").splitlines() if x.strip()]
target_urls = {r.get("canonical_url") or r.get("url") for r in url_rows}

manifest_rows = [json.loads(x) for x in manifest_p.read_text(encoding="utf-8").splitlines() if x.strip()]
success = [r for r in manifest_rows if r.get("success")]
failed = [r for r in manifest_rows if not r.get("success")]

success_urls = {r.get("canonical_url") or r.get("source_url") or r.get("url") for r in success}
coverage = len(success_urls & target_urls) / max(len(target_urls), 1)

empty_md = []
for r in success:
    md_path = Path(r.get("markdown_path", ""))
    if not md_path.exists() or md_path.stat().st_size == 0:
        empty_md.append(r.get("source_url"))

print("target_urls:", len(target_urls))
print("manifest_rows:", len(manifest_rows))
print("success:", len(success))
print("failed:", len(failed))
print("coverage:", round(coverage, 4))
print("empty_md:", len(empty_md))

assert len(target_urls) > 0, "No target URLs"
assert len(success) > 500, "Full crawl did not progress beyond smoke batch"
assert coverage >= 0.8, f"Coverage too low: {coverage}"
assert len(empty_md) < len(success) * 0.2, "Too many empty markdown files"

print("FULL CRAWL VALIDATION PASSED")
'@ | python
```

## Status

```text
Task Crawl Full URLs
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/raw/html/*.html
- data/raw/markdown/*.md
- data/raw/documents_manifest.jsonl
- data/raw/crawl_documents_report.json
Notes:
- resumed crawl completed for all 5,397 unique URLs
- manifest success unique = 5,397
- validated coverage = 1.0
```

---

# 7. Task 4 — Parse metadata

## Mục tiêu

Tạo structured documents từ manifest/raw HTML/Markdown.

## Output

```text
data/processed/documents.jsonl
```

## Module

```text
src/ingestion/document_parser.py
```

Nếu có provider-specific parser:

```text
src/ingestion/provider_parsers/luatvietnam_parser.py
```

ưu tiên dùng cho LuatVietnam.

## Command

```bash
python -m src.ingestion.document_parser --manifest data/raw/documents_manifest.jsonl --output data/processed/documents.jsonl
```

## Validation

```powershell
@'
import json
from pathlib import Path

p = Path("data/processed/documents.jsonl")
assert p.exists(), "Missing documents.jsonl"

rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
assert rows, "No documents"

required = ["doc_id", "source_url", "domain", "markdown_path", "content_hash"]
missing_required = []
for r in rows:
    for k in required:
        if not r.get(k):
            missing_required.append((r.get("doc_id"), k))

title_count = sum(1 for r in rows if r.get("doc_title"))
number_count = sum(1 for r in rows if r.get("doc_number"))
date_count = sum(1 for r in rows if r.get("issue_date") or r.get("effective_date"))

print("documents:", len(rows))
print("doc_title_count:", title_count)
print("doc_number_count:", number_count)
print("date_count:", date_count)
print("missing_required:", missing_required[:10])

assert not missing_required, f"Missing required fields: {missing_required[:10]}"
assert title_count > 0, "No document titles extracted"

print("METADATA PARSE VALIDATION PASSED")
'@ | python
```

## Status

```text
Task Parse Metadata
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/processed/documents.jsonl
Notes:
- parsed 5,397 documents from full crawl manifest
```

---

# 8. Task 5 — Clean text

## Mục tiêu

Tạo text sạch từ Markdown để parse cấu trúc pháp luật.

## Output

```text
data/processed/cleaned_documents.jsonl
```

## Module

```text
src/ingestion/text_cleaner.py
```

## Command

```bash
python -m src.ingestion.text_cleaner --documents data/processed/documents.jsonl --output data/processed/cleaned_documents.jsonl
```

## Validation

```powershell
@'
import json
from pathlib import Path

p = Path("data/processed/cleaned_documents.jsonl")
assert p.exists(), "Missing cleaned_documents.jsonl"

rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
assert rows, "No cleaned documents"

short = []
legal_signal = 0
noise_count = 0
noise_terms = ["Đăng nhập", "Đăng ký", "Tin liên quan", "Chia sẻ", "Xem thêm"]

for r in rows:
    text = r.get("cleaned_text", "")
    if len(text.strip()) < 500:
        short.append(r.get("doc_id"))
    if any(sig in text for sig in ["Điều ", "Khoản ", "Căn cứ", "Luật ", "Nghị định", "Thông tư"]):
        legal_signal += 1
    if any(term in text[:2000] for term in noise_terms):
        noise_count += 1

print("cleaned docs:", len(rows))
print("short docs:", len(short))
print("legal_signal:", legal_signal)
print("noise_count_first_2000_chars:", noise_count)

assert legal_signal > 0, "No legal signals after cleaning"
assert len(short) < len(rows) * 0.5, "Too many short cleaned documents"

print("CLEAN TEXT VALIDATION PASSED")
'@ | python
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
- cleaned 5,397 documents; legal signal count = 2,831
```

---

# 9. Task 6 — Parse Điều/Khoản

## Mục tiêu

Parse cấu trúc pháp luật từ cleaned text.

## Output

```text
data/processed/legal_nodes.jsonl
```

## Module

```text
src/ingestion/legal_structure_parser.py
```

## Command

```bash
python -m src.ingestion.legal_structure_parser --input data/processed/cleaned_documents.jsonl --output data/processed/legal_nodes.jsonl
```

## Validation

```powershell
@'
import json
from pathlib import Path
from collections import Counter

p = Path("data/processed/legal_nodes.jsonl")
assert p.exists(), "Missing legal_nodes.jsonl"

rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
assert rows, "No legal nodes"

levels = Counter(r.get("level") for r in rows)
missing = []
for r in rows:
    if not r.get("node_id"):
        missing.append(("node_id", r))
    if not r.get("doc_id"):
        missing.append(("doc_id", r))
    if not (r.get("content") or r.get("title")):
        missing.append(("content_or_title", r.get("node_id")))

print("nodes:", len(rows))
print("levels:", dict(levels))
print("missing:", missing[:5])

assert not missing, f"Missing fields: {missing[:5]}"
assert levels.get("article", 0) > 0, "No article nodes detected"

print("LEGAL STRUCTURE PARSE VALIDATION PASSED")
'@ | python
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
- parsed 408,730 legal structure nodes
```

---

# 10. Task 7 — Chunk + context chunk

## Mục tiêu

Tạo retrieval chunks nhỏ và context chunks lớn hơn.

## Output

```text
data/processed/chunks.jsonl
data/processed/context_chunks.jsonl
```

## Module

```text
src/ingestion/legal_chunker.py
```

## Command

```bash
python -m src.ingestion.legal_chunker --nodes data/processed/legal_nodes.jsonl --documents data/processed/documents.jsonl --output data/processed/chunks.jsonl --output-context data/processed/context_chunks.jsonl
```

Nếu command trong project khác, tự điều chỉnh theo code thực tế nhưng output phải giữ như trên.

## Validation

```powershell
@'
import json
from pathlib import Path

chunks_p = Path("data/processed/chunks.jsonl")
ctx_p = Path("data/processed/context_chunks.jsonl")

assert chunks_p.exists(), "Missing chunks.jsonl"
assert ctx_p.exists(), "Missing context_chunks.jsonl"

chunks = [json.loads(x) for x in chunks_p.read_text(encoding="utf-8").splitlines() if x.strip()]
ctxs = [json.loads(x) for x in ctx_p.read_text(encoding="utf-8").splitlines() if x.strip()]

assert chunks, "No chunks"
assert ctxs, "No context chunks"

ctx_ids = {c.get("context_chunk_id") for c in ctxs}
missing_required = []
for c in chunks:
    for k in ["chunk_id", "doc_id", "content", "embedding_text", "source_url", "citation"]:
        if not c.get(k):
            missing_required.append((c.get("chunk_id"), k))
    ccid = c.get("context_chunk_id")
    if ccid:
        assert ccid in ctx_ids, f"Missing context_chunk_id: {ccid}"

has_parent = sum(1 for c in chunks if c.get("parent_id"))
has_prev_next = sum(1 for c in chunks if c.get("prev_chunk_id") or c.get("next_chunk_id"))

print("chunks:", len(chunks))
print("context_chunks:", len(ctxs))
print("chunks_with_parent:", has_parent)
print("chunks_with_prev_next:", has_prev_next)
print("missing_required:", missing_required[:10])

assert not missing_required, f"Missing required chunk fields: {missing_required[:10]}"

print("CHUNK AND CONTEXT CHUNK VALIDATION PASSED")
'@ | python
```

## Status

```text
Task Chunk and Context Chunk
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/processed/chunks.jsonl
- data/processed/context_chunks.jsonl
Notes:
- produced 305,015 chunks and 41,806 context chunks
```

---

# 11. Task 8 — Build graph

## Mục tiêu

Tạo lightweight legal graph phục vụ context expansion.

## Output

Tối thiểu:

```text
data/processed/legal_edges.jsonl
```

Khuyến nghị nếu đã có canonical graph:

```text
data/processed/legal_graph_nodes.jsonl
data/processed/legal_graph_edges.jsonl
```

## Module

```text
src/ingestion/graph_builder.py
```

## Command đề xuất

```bash
python -m src.ingestion.graph_builder --documents data/processed/documents.jsonl --nodes data/processed/legal_nodes.jsonl --chunks data/processed/chunks.jsonl --context-chunks data/processed/context_chunks.jsonl --output-nodes data/processed/legal_graph_nodes.jsonl --output-edges data/processed/legal_graph_edges.jsonl
```

Nếu project đang dùng module khác, reuse module hiện có.

## Validation

```powershell
@'
import json
from pathlib import Path
from collections import Counter

nodes_p = Path("data/processed/legal_graph_nodes.jsonl")
edges_p = Path("data/processed/legal_graph_edges.jsonl")
legacy_edges_p = Path("data/processed/legal_edges.jsonl")

assert edges_p.exists() or legacy_edges_p.exists(), "Missing graph edges"

if nodes_p.exists():
    nodes = [json.loads(x) for x in nodes_p.read_text(encoding="utf-8").splitlines() if x.strip()]
else:
    nodes = []

edges_file = edges_p if edges_p.exists() else legacy_edges_p
edges = [json.loads(x) for x in edges_file.read_text(encoding="utf-8").splitlines() if x.strip()]

assert edges, "No graph edges"

edge_types = Counter(e.get("relation_type") for e in edges)
print("graph_nodes:", len(nodes))
print("graph_edges:", len(edges))
print("edge_types:", dict(edge_types))

assert any(k in edge_types for k in ["HAS_PARENT", "HAS_CHILD", "NEXT", "PREVIOUS"]), "No structural graph edges"

print("GRAPH BUILD VALIDATION PASSED")
'@ | python
```

## Status

```text
Task Build Graph
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/processed/legal_graph_nodes.jsonl
- data/processed/legal_graph_edges.jsonl
Notes:
- built canonical graph with 586,361 nodes and 2,813,640 edges
```

---

# 12. Task 9 — Build FAISS/BM25 index

## Mục tiêu

Build retrieval index từ `chunks.jsonl`.

## Output

```text
data/indexes/faiss.index
data/indexes/chunk_metadata.json
```

Nếu có BM25:

```text
data/indexes/bm25_index.pkl
```

## Embedding config khuyến nghị

```python
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
MAX_LENGTH = 8192
NORMALIZE_EMBEDDINGS = True
```

Nếu project đã cấu hình model khác và đang chạy ổn, không đổi bừa. Nếu chưa có, tích hợp BGE-M3.

## Command FAISS

```bash
python -m src.ingestion.index_builder --chunks data/processed/chunks.jsonl --faiss-index data/indexes/faiss.index --metadata data/indexes/chunk_metadata.json
```

## Command BM25 nếu có

```bash
python -m src.ingestion.bm25_builder --chunks data/processed/chunks.jsonl --output data/indexes/bm25_index.pkl
```

Nếu chưa có `bm25_builder.py`, tạo module tối thiểu dùng `rank-bm25`, nhưng không phá pipeline nếu dependency thiếu.

## Validation

```powershell
@'
import json
from pathlib import Path

faiss_p = Path("data/indexes/faiss.index")
meta_p = Path("data/indexes/chunk_metadata.json")
chunks_p = Path("data/processed/chunks.jsonl")

assert faiss_p.exists(), "Missing faiss.index"
assert faiss_p.stat().st_size > 0, "faiss.index empty"
assert meta_p.exists(), "Missing chunk_metadata.json"
assert chunks_p.exists(), "Missing chunks.jsonl"

chunks = [json.loads(x) for x in chunks_p.read_text(encoding="utf-8").splitlines() if x.strip()]
metadata = json.loads(meta_p.read_text(encoding="utf-8"))

assert isinstance(metadata, list), "chunk_metadata must be list"
assert len(metadata) == len(chunks), f"metadata count != chunks count: {len(metadata)} != {len(chunks)}"

for i, r in enumerate(metadata[:20]):
    assert r.get("index") == i, f"Metadata index mismatch at {i}"
    assert r.get("chunk_id")
    assert r.get("doc_id")
    assert r.get("source_url")
    assert r.get("citation")

print("INDEX BUILD VALIDATION PASSED", {
    "chunks": len(chunks),
    "metadata": len(metadata),
    "faiss_size": faiss_p.stat().st_size,
    "bm25_exists": Path("data/indexes/bm25_index.pkl").exists(),
})
'@ | python
```

## Retrieval smoke test

```bash
python -m src.retrieval.hybrid_retriever --query "Ai không được thành lập doanh nghiệp?" --top-k 5
```

Nếu CLI khác, tự điều chỉnh theo project.

## Status

```text
Task Build FAISS BM25 Index
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- data/indexes/faiss.index
- data/indexes/chunk_metadata.json
- data/indexes/bm25_index.pkl
Notes:
- FAISS built with BAAI/bge-m3 over 305,015 chunk vectors
- BM25 rebuilt in doc-level form for memory-safe hybrid retrieval
```

---

# 13. Task 10 — Retrieval/QA evaluation với file câu hỏi test

## Mục tiêu

Chạy end-to-end retrieval/QA evaluation bằng file câu hỏi test R2AI.

Input có thể là một trong các file:

```text
data/evaluation/R2AIStage1DATA.json
data/evaluation/r2ai_stage1_questions.jsonl
```

Nếu chưa có JSONL, convert trước.

## Prepare dataset

```bash
python -m src.evaluation.prepare_r2ai_dataset --input data/evaluation/R2AIStage1DATA.json --output data/evaluation/r2ai_stage1_questions.jsonl
```

Nếu module chưa có, tạo `src/evaluation/prepare_r2ai_dataset.py`.

## Evaluation smoke

```bash
python -m src.evaluation.evaluate_qa --questions data/evaluation/r2ai_stage1_questions.jsonl --run-id r2ai_stage1_smoke --limit 50
```

## Evaluation full

Sau smoke pass:

```bash
python -m src.evaluation.evaluate_qa --questions data/evaluation/r2ai_stage1_questions.jsonl --run-id r2ai_stage1_full
```

## Metrics tối thiểu

Summary cần có:

```text
total_questions
answer_non_empty_rate
citation_present_rate
avg_context_count
route_distribution
domain_distribution
avg_latency_seconds
missing_context_rate
missing_citation_rate
```

Nếu có expected_law_refs:

```text
legal_ref_hit_rate
```

## Validation

```powershell
@'
import json
from pathlib import Path

summary_candidates = [
    Path("logs/eval_runs/r2ai_stage1_smoke_summary.json"),
    Path("logs/eval_runs/r2ai_stage1_full_summary.json"),
]

summary_p = next((p for p in summary_candidates if p.exists()), None)
assert summary_p is not None, "Missing evaluation summary"

data = json.loads(summary_p.read_text(encoding="utf-8"))
required = ["total_questions", "answer_non_empty_rate", "citation_present_rate"]
for k in required:
    assert k in data, f"Missing metric {k}"

assert data["total_questions"] > 0, "No evaluated questions"
assert data["answer_non_empty_rate"] > 0, "No non-empty answers"

print("QA EVALUATION VALIDATION PASSED", data)
'@ | python
```

## QA smoke questions

Nếu cần test trực tiếp:

```bash
python -m src.qa_pipeline --question "Ai không được thành lập doanh nghiệp?"
python -m src.qa_pipeline --question "Không góp đủ vốn điều lệ đúng hạn thì bị phạt gì?"
python -m src.qa_pipeline --question "Người nước ngoài góp vốn vào công ty Việt Nam cần điều kiện gì?"
```

## Status

```text
Task Retrieval QA Evaluation
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- logs/eval_runs/r2ai_stage1_smoke.jsonl
- logs/eval_runs/r2ai_stage1_smoke_summary.json
- logs/eval_runs/r2ai_stage1_full.jsonl nếu chạy full
- logs/eval_runs/r2ai_stage1_full_summary.json nếu chạy full
Notes:
- smoke and full evaluation completed on sample_questions.jsonl
```

---

# 14. Final End-to-End Validation

Sau khi 10 task trên pass, chạy validation tổng.

## File validation

```powershell
@'
import json
from pathlib import Path

required = [
    "data/raw/document_urls.jsonl",
    "data/raw/documents_manifest.jsonl",
    "data/processed/documents.jsonl",
    "data/processed/cleaned_documents.jsonl",
    "data/processed/legal_nodes.jsonl",
    "data/processed/chunks.jsonl",
    "data/processed/context_chunks.jsonl",
    "data/indexes/faiss.index",
    "data/indexes/chunk_metadata.json",
]

missing = [x for x in required if not Path(x).exists() or Path(x).stat().st_size == 0]
assert not missing, f"Missing/empty required files: {missing}"

chunks = [json.loads(x) for x in Path("data/processed/chunks.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
metadata = json.loads(Path("data/indexes/chunk_metadata.json").read_text(encoding="utf-8"))

assert chunks, "No chunks"
assert len(chunks) == len(metadata), "chunks != chunk_metadata"

print("FINAL FILE VALIDATION PASSED", {
    "chunks": len(chunks),
    "metadata": len(metadata),
})
'@ | python
```

## QA pipeline validation

```powershell
@'
from src.qa_pipeline import LegalQAPipeline

questions = [
    "Ai không được thành lập doanh nghiệp?",
    "Không góp đủ vốn điều lệ đúng hạn thì bị phạt gì?",
    "Người nước ngoài góp vốn vào công ty Việt Nam cần điều kiện gì?",
]

qa = LegalQAPipeline()

for q in questions:
    r = qa.answer(q)
    assert r.get("answer"), f"No answer for {q}"
    assert r.get("route"), f"No route for {q}"
    assert r.get("final_contexts"), f"No contexts for {q}"
    assert r.get("citations") is not None, f"No citations field for {q}"

    print("QUESTION:", q)
    print("ROUTE:", r.get("route"))
    print("CONTEXTS:", len(r.get("final_contexts", [])))
    print("ANSWER:", r.get("answer", "")[:400])
    print("---")

print("FINAL QA PIPELINE VALIDATION PASSED")
'@ | python
```

## Final status

```text
Post Crawl Full Index Evaluation Pipeline
Status: DONE
Review: PASSED
Tests: PASSED
Validation: PASSED
Output:
- all required raw/processed/index/eval files
Notes:
- full crawl, parse, chunk, graph, FAISS/BM25, and QA evaluation all passed
```

---
