# CODEX TASK SPEC — Refactor Multi-turnRAG for R2AI Legal AI Assistant

> Mục tiêu: chuyển project `Multi-turnRAG` từ chatbot RAG thử nghiệm nhiều model sang một codebase gọn, dễ teamwork, dễ cải tiến và phù hợp với cuộc thi **R2AI2026 Build Legal AI Assistant**.
>
> Nguyên tắc bắt buộc: **chỉ chuyển sang task tiếp theo khi task hiện tại đã hoàn thành, được review, test thành công và cập nhật trạng thái trong file này**.

---

## 0. Bối cảnh cuộc thi cần bám sát

Cuộc thi yêu cầu hệ thống Legal AI Assistant có khả năng:

1. Truy hồi đúng văn bản pháp luật / điều luật liên quan.
2. Sinh câu trả lời pháp luật bằng tiếng Việt.
3. Dẫn nguồn rõ ràng theo `Điều X`, tên văn bản và mã văn bản.
4. Hạn chế hallucination, không bịa điều luật.
5. Xuất bài nộp đúng format `results.json`.

Format bài nộp cần có dạng:

```json
[
  {
    "id": 1,
    "question": "...",
    "answer": "...",
    "relevant_docs": ["<mã văn bản>|<tên văn bản>"],
    "relevant_articles": ["<mã văn bản>|<tên văn bản>|<điều>"]
  }
]
```

Lưu ý quan trọng:

- Hệ thống chấm tự động có thể trích pattern `Điều X` trong trường `answer`.
- Vì vậy `answer` phải nhắc rõ căn cứ pháp lý, ví dụ: `Căn cứ Điều 4 Luật ...`.
- Evaluation retrieval cần chuyển từ `Hit@5 / Recall@5 / MRR` sang thêm `Precision / Recall / F2 macro` ở cấp **article-level**.
- Project cũ có nhiều module multi-turn/history/ablation. Giữ lại phần có ích, xóa hoặc cô lập phần thừa để dễ teamwork.

---

## 1. Quy tắc làm việc bắt buộc cho Codex

### 1.1. Không làm nhiều task lớn cùng lúc

Codex phải làm theo thứ tự:

1. **Task 1 — Cleanup & Teamwork Project Structure**
2. **Task 2 — Competition Legal IR & QA Layer**

Không được bắt đầu Task 2 nếu Task 1 chưa có trạng thái `DONE`, `REVIEWED`, `TESTED`.

### 1.2. Mỗi task phải có đủ 5 bước

Với mỗi task, thực hiện đúng quy trình:

```text
PLAN → IMPLEMENT → SELF-REVIEW → TEST → UPDATE STATUS
```

Trong đó:

- `PLAN`: liệt kê file sẽ sửa/xóa/tạo.
- `IMPLEMENT`: chỉnh code.
- `SELF-REVIEW`: rà lại logic, naming, compatibility.
- `TEST`: chạy test hoặc script kiểm tra tối thiểu.
- `UPDATE STATUS`: cập nhật bảng trạng thái trong file này.

### 1.3. Không xóa code khi chưa chắc

Nếu gặp file/module chưa rõ còn dùng hay không:

- Không xóa ngay.
- Di chuyển vào thư mục `legacy/` hoặc ghi chú trong `docs/legacy_notes.md`.
- Chỉ xóa khi đã xác nhận không được import, không được dùng trong CLI, test, pipeline hoặc README mới.

### 1.4. Mỗi thay đổi lớn cần có log

Sau mỗi task, cập nhật:

```text
CHANGELOG.md
```

với nội dung:

```md
## YYYY-MM-DD — Task X
- Changed:
- Removed:
- Added:
- Tests:
- Notes:
```

---

## 2. Bảng trạng thái tổng

| Task | Tên task | Status | Review | Test | Ghi chú |
|---|---|---|---|---|---|
| Task 1 | Cleanup & Teamwork Project Structure | DONE | REVIEWED | TESTED_PASS | Audit xong, tái cấu trúc xong, smoke test và pytest pass |
| Task 2 | Competition Legal IR & QA Layer | DONE | REVIEWED | TESTED_PASS | Metadata, aggregation, answer, submission và evaluation đã pass sample end-to-end |

Quy ước trạng thái:

```text
TODO       = chưa làm
IN_PROGRESS = đang làm
DONE       = code đã xong
REVIEWED   = đã self-review
TESTED     = đã test pass
BLOCKED    = bị chặn bởi task trước hoặc lỗi chưa xử lý
```

---

# TASK 1 — Cleanup & Teamwork Project Structure

## 1.1. Mục tiêu

Dọn project cũ để codebase dễ hiểu, dễ teamwork, dễ cải tiến. Tập trung loại bỏ hoặc cô lập các phần thừa từ project `Multi-turnRAG` cũ, đặc biệt các phần không cần cho bài thi batch Legal IR & QA.

## 1.2. Phạm vi cần làm

### A. Audit toàn bộ project

Tạo file:

```text
docs/project_audit.md
```

Nội dung cần có:

```md
# Project Audit

## Active modules
- module/file:
- purpose:
- used by:

## Legacy / unused candidates
- module/file:
- reason:
- action: keep / move_to_legacy / remove

## Risky dependencies
- dependency:
- reason:
- replacement plan:
```

Cần kiểm tra tối thiểu:

- `src/`
- `scripts/`
- `notebooks/`
- `data/`
- `eval/` hoặc `evaluation/`
- `configs/`
- `README.md`
- file `.env.example`
- dependency files: `requirements.txt`, `pyproject.toml`, `environment.yml` nếu có.

### B. Tổ chức lại cấu trúc project

Đề xuất cấu trúc mới:

```text
Multi-turnRAG/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── .env.example
├── configs/
│   ├── retrieval.yaml
│   ├── generation.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── indexes/
│   └── submissions/
├── docs/
│   ├── project_audit.md
│   ├── architecture.md
│   ├── metadata_schema.md
│   ├── answer_generation.md
│   └── evaluation.md
├── legacy/
│   └── README.md
├── scripts/
│   ├── build_corpus.py
│   ├── build_index.py
│   ├── run_retrieval.py
│   ├── generate_submission.py
│   ├── validate_submission.py
│   └── evaluate_submission.py
├── src/
│   └── legal_rag/
│       ├── __init__.py
│       ├── config/
│       ├── corpus/
│       ├── retrieval/
│       ├── reranking/
│       ├── aggregation/
│       ├── generation/
│       ├── evaluation/
│       ├── submission/
│       └── utils/
└── tests/
    ├── test_metadata_schema.py
    ├── test_article_aggregation.py
    ├── test_submission_format.py
    └── test_evaluation_metrics.py
```

Có thể điều chỉnh theo codebase thật, nhưng phải đảm bảo:

- module retrieval tách khỏi generation;
- module evaluation tách khỏi app/demo;
- module submission tách riêng;
- legacy code không trộn với code chính;
- config tách khỏi source code.

### C. Xóa hoặc cô lập phần thừa từ project cũ

Các phần có thể đưa vào `legacy/` nếu không phục vụ submission:

- UI chatbot cũ nếu không cần cho batch submission.
- multi-turn history selection nếu test set chỉ single-turn.
- notebook thử nghiệm không còn dùng.
- script ablation cũ nếu không còn chạy được.
- code gọi closed API model nếu không hợp lệ với rule cuộc thi.
- file output/cache cũ, log cũ, index cũ không tái lập được.

Không xóa trực tiếp nếu chưa chắc. Ưu tiên:

```text
move_to_legacy → test import → remove later if safe
```

### D. Chuẩn hóa môi trường teamwork

Cần tạo hoặc cập nhật:

```text
README.md
.env.example
CHANGELOG.md
.gitignore
```

README mới cần có:

```md
# R2AI Legal AI Assistant

## Project purpose
## Architecture overview
## Setup
## Build corpus
## Build index
## Run retrieval
## Generate submission
## Validate submission
## Evaluate locally
## Team workflow
## Coding conventions
```

`.env.example` cần có biến môi trường mẫu, không chứa secret thật:

```env
EMBEDDING_MODEL=intfloat/multilingual-e5-base
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
LLM_MODEL=Qwen2.5-7B-Instruct
VECTOR_INDEX_PATH=data/indexes/faiss.index
METADATA_PATH=data/processed/articles.jsonl
SUBMISSION_PATH=data/submissions/results.json
```

### E. Chuẩn hóa import path

Sau khi reorganize, code chính nên import theo package:

```python
from legal_rag.retrieval.hybrid import HybridRetriever
from legal_rag.aggregation.article import ArticleAggregator
from legal_rag.submission.schema import SubmissionItem
```

Tránh import tương đối lộn xộn hoặc import file script trực tiếp.

## 1.3. Acceptance Criteria cho Task 1

Task 1 chỉ được coi là xong khi:

- [ ] Có `docs/project_audit.md`.
- [ ] Có cấu trúc thư mục rõ ràng cho teamwork.
- [ ] Có `legacy/README.md` giải thích những gì được chuyển vào legacy.
- [ ] README được cập nhật theo mục tiêu R2AI Legal AI Assistant.
- [ ] `.env.example` không chứa secret.
- [ ] Import package chính không lỗi.
- [ ] Không còn file output/cache rác trong root project.
- [ ] Chạy được ít nhất một lệnh smoke test.
- [ ] Cập nhật `CHANGELOG.md`.
- [ ] Cập nhật bảng trạng thái Task 1 trong file này.

## 1.4. Smoke test bắt buộc cho Task 1

Chạy tối thiểu:

```bash
python -m compileall src scripts
```

Nếu có pytest:

```bash
pytest -q
```

Nếu chưa có pytest ổn định, tạo ít nhất test import:

```bash
python - <<'PY'
import legal_rag
print('legal_rag import OK')
PY
```

## 1.5. Trạng thái Task 1

| Step | Status | Notes |
|---|---|---|
| PLAN | DONE | Audit repo, chốt active vs legacy, dựng cấu trúc teamwork mới |
| IMPLEMENT | DONE | Tạo package `legal_rag`, docs/config mới, cô lập legacy và chuẩn hóa path |
| SELF-REVIEW | REVIEWED | Rà lại import path, default path, fallback splitter và compatibility layer |
| TEST | TESTED_PASS | `compileall`, import smoke test, `pytest -q` đều pass |
| UPDATE STATUS | DONE | Đã cập nhật bảng trạng thái và changelog |

---

# TASK 2 — Competition Legal IR & QA Layer

## 2.1. Điều kiện mở task

Chỉ bắt đầu Task 2 nếu Task 1 đạt:

```text
Status: DONE
Review: REVIEWED
Test: TESTED
```

Nếu chưa đạt, dừng lại và hoàn thiện Task 1 trước.

## 2.2. Mục tiêu

Thêm lớp phục vụ trực tiếp cuộc thi:

1. Format lại metadata ở cấp văn bản và điều luật.
2. Chuẩn hóa cách sinh answer có căn cứ pháp lý.
3. Thêm article-level aggregation.
4. Thêm evaluation theo Precision / Recall / F2 macro.
5. Thêm script tạo và validate `results.json`.

## 2.3. Metadata schema mới

Tạo file tài liệu:

```text
docs/metadata_schema.md
```

Tạo module:

```text
src/legal_rag/corpus/schema.py
```

Schema đề xuất cho một article/chunk:

```python
from pydantic import BaseModel, Field
from typing import Optional

class LegalArticle(BaseModel):
    doc_id: str = Field(..., description='Mã văn bản, ví dụ 04/2017/QH14')
    doc_title: str = Field(..., description='Tên văn bản ngắn')
    doc_full_name: str = Field(..., description='Loại văn bản + mã văn bản + trích yếu')
    article_id: str = Field(..., description='Định danh đầy đủ: doc_id|doc_full_name|Điều X')
    article_number: str = Field(..., description='Điều X')
    article_title: Optional[str] = None
    clause_number: Optional[str] = None
    chunk_id: str
    chunk_text: str
    source_path: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
```

Yêu cầu chuẩn hóa:

- `article_id` phải đúng format submission:

```text
<mã văn bản>|<tên văn bản>|<điều>
```

- `doc_ref` phải đúng format:

```text
<mã văn bản>|<tên văn bản>
```

- `article_number` phải thống nhất dạng:

```text
Điều 1
Điều 2
Điều 4
Điều 10
```

Không dùng lẫn:

```text
Article 4
D.4
Điều thứ 4
ĐIỀU 4
```

## 2.4. Article-level aggregation

Tạo module:

```text
src/legal_rag/aggregation/article.py
```

Mục tiêu: gom nhiều chunk retrieval về cùng một điều luật.

Input:

```python
retrieved_chunks: list[RetrievedChunk]
```

Output:

```python
selected_articles: list[SelectedArticle]
```

Logic đề xuất:

1. Group theo `article_id`.
2. Score article bằng một trong các cách:
   - max chunk score;
   - weighted sum top chunks;
   - rerank score nếu có;
   - kết hợp dense_score, bm25_score, rerank_score.
3. Loại duplicate.
4. Sort giảm dần theo score.
5. Chọn top-k hoặc dynamic threshold.

Config:

```yaml
article_selection:
  strategy: dynamic_threshold
  default_top_k: 5
  min_articles: 1
  max_articles: 7
  score_threshold: 0.35
  relative_threshold: 0.75
```

Dynamic top-k gợi ý:

```text
- Nếu top1 score cao và gap top1-top2 lớn → chọn 1-2 article.
- Nếu nhiều article score gần nhau → chọn 3-5 article.
- Nếu câu hỏi dạng điều kiện/quy trình/nghĩa vụ → cho phép chọn nhiều hơn.
```

## 2.5. Answer generation

Tạo tài liệu:

```text
docs/answer_generation.md
```

Tạo module:

```text
src/legal_rag/generation/answer_generator.py
```

Yêu cầu answer:

- Trả lời bằng tiếng Việt.
- Có kết luận ngắn ở đầu.
- Bắt buộc nhắc rõ `Điều X` và tên văn bản.
- Chỉ dùng căn cứ được cung cấp.
- Không bịa điều luật.
- Nếu thiếu căn cứ, nói rõ giới hạn.
- Có phần áp dụng thực tế cho SME.
- Dễ hiểu với người không chuyên.

Template đề xuất:

```text
Bạn là trợ lý pháp lý cho doanh nghiệp SME tại Việt Nam.
Chỉ trả lời dựa trên các căn cứ pháp luật được cung cấp.
Không được bịa điều luật, số điều, tên văn bản hoặc điều kiện pháp lý ngoài căn cứ.

Câu hỏi:
{question}

Căn cứ pháp lý:
{evidence_pack}

Yêu cầu trả lời theo cấu trúc:
1. Kết luận ngắn: trả lời trực tiếp câu hỏi.
2. Căn cứ pháp lý: nhắc rõ Điều X và tên văn bản.
3. Giải thích dễ hiểu: diễn giải quy định bằng ngôn ngữ đơn giản.
4. Gợi ý áp dụng thực tế: doanh nghiệp cần kiểm tra/làm gì.
5. Lưu ý giới hạn: đây là tư vấn sơ bộ dựa trên căn cứ được cung cấp.
```

Output answer nên có dạng:

```text
Kết luận ngắn: ...

Căn cứ pháp lý: Căn cứ Điều 4 Luật ... và Điều 5 Nghị định ..., ...

Giải thích dễ hiểu: ...

Gợi ý áp dụng thực tế: ...

Lưu ý: Đây là tư vấn sơ bộ dựa trên các căn cứ được cung cấp; doanh nghiệp nên kiểm tra hồ sơ cụ thể hoặc hỏi chuyên gia pháp lý khi cần quyết định chính thức.
```

## 2.6. Citation validator

Tạo module:

```text
src/legal_rag/generation/citation_validator.py
```

Chức năng:

1. Dùng regex trích các pattern `Điều X` trong answer.
2. So sánh với `selected_articles`.
3. Báo lỗi nếu answer không chứa điều luật nào.
4. Báo cảnh báo nếu answer nhắc điều không nằm trong selected articles.
5. Có thể tự sửa nhẹ bằng cách thêm câu căn cứ pháp lý nếu thiếu.

Regex gợi ý:

```python
ARTICLE_PATTERN = r"Điều\s+\d+[a-zA-ZăâêôơưĂÂÊÔƠƯĐđ]*"
```

Validation result:

```python
class CitationValidationResult(BaseModel):
    ok: bool
    cited_articles: list[str]
    missing_articles: list[str]
    unsupported_citations: list[str]
    warnings: list[str]
```

## 2.7. Submission schema và exporter

Tạo module:

```text
src/legal_rag/submission/schema.py
src/legal_rag/submission/exporter.py
src/legal_rag/submission/validator.py
```

Schema:

```python
from pydantic import BaseModel

class SubmissionItem(BaseModel):
    id: int
    question: str
    answer: str
    relevant_docs: list[str]
    relevant_articles: list[str]
```

Exporter cần tạo file:

```text
data/submissions/results.json
```

Validator kiểm tra:

- JSON là list.
- Mỗi item có đủ field.
- `id` là int.
- `question`, `answer` là string không rỗng.
- `relevant_docs` là list string đúng format 2 phần.
- `relevant_articles` là list string đúng format 3 phần.
- Không duplicate `id`.
- Không duplicate article trong cùng một item.
- Answer có ít nhất một pattern `Điều X` nếu `relevant_articles` không rỗng.
- Các article trong `relevant_articles` xuất hiện hợp lý trong answer.

Script CLI:

```text
scripts/generate_submission.py
scripts/validate_submission.py
```

Ví dụ chạy:

```bash
python scripts/generate_submission.py \
  --input data/raw/test_questions.json \
  --output data/submissions/results.json \
  --config configs/retrieval.yaml

python scripts/validate_submission.py \
  --input data/submissions/results.json
```

## 2.8. Evaluation theo Precision / Recall / F2 macro

Tạo tài liệu:

```text
docs/evaluation.md
```

Tạo module:

```text
src/legal_rag/evaluation/metrics.py
src/legal_rag/evaluation/evaluator.py
```

Input:

```text
predictions: results.json
gold: gold_articles.json hoặc dev set tự tạo
```

Metric cần có:

```python
def precision(pred: set[str], gold: set[str]) -> float:
    if not pred:
        return 0.0
    return len(pred & gold) / len(pred)


def recall(pred: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(pred & gold) / len(gold)


def f2_score(p: float, r: float) -> float:
    if p == 0 and r == 0:
        return 0.0
    return (5 * p * r) / (4 * p + r)
```

Macro average:

```python
macro_precision = average(per_question_precision)
macro_recall = average(per_question_recall)
macro_f2 = average(per_question_f2)
```

Evaluator output:

```json
{
  "macro_precision": 0.0,
  "macro_recall": 0.0,
  "macro_f2": 0.0,
  "num_questions": 0,
  "details": [
    {
      "id": 1,
      "precision": 0.0,
      "recall": 0.0,
      "f2": 0.0,
      "predicted_articles": [],
      "gold_articles": [],
      "correct_articles": []
    }
  ]
}
```

Script CLI:

```text
scripts/evaluate_submission.py
```

Ví dụ chạy:

```bash
python scripts/evaluate_submission.py \
  --pred data/submissions/results.json \
  --gold data/processed/dev_gold.json \
  --output data/submissions/eval_report.json
```

## 2.9. QA quality checker

Tạo module tùy chọn nhưng nên có:

```text
src/legal_rag/generation/quality_checker.py
```

Rule-based checker, không cần LLM judge.

Check các tiêu chí:

### Căn cứ chính xác pháp luật

- Có `Điều X` trong answer.
- Điều được cite nằm trong selected articles.

### Tính chính xác nội dung

- Không có câu khẳng định nếu evidence rỗng.
- Không có cụm mơ hồ như `theo luật hiện hành` mà không cite.
- Không có tên luật không nằm trong evidence.

### Tính đầy đủ & toàn diện

- Nếu câu hỏi có từ khóa `điều kiện`, `thủ tục`, `nghĩa vụ`, `quyền`, `xử phạt`, kiểm tra số lượng article tối thiểu.

### Tính thực tiễn

- Answer có phần `Gợi ý áp dụng thực tế` hoặc câu tương đương.

### Tính rõ ràng

- Answer có cấu trúc rõ: kết luận, căn cứ, giải thích, lưu ý.
- Câu trả lời không quá ngắn nếu có nhiều căn cứ.

Output:

```python
class AnswerQualityReport(BaseModel):
    ok: bool
    legal_basis_ok: bool
    content_warnings: list[str]
    completeness_warnings: list[str]
    practicality_ok: bool
    clarity_ok: bool
```

## 2.10. Acceptance Criteria cho Task 2

Task 2 chỉ được coi là xong khi:

- [ ] Có metadata schema chuẩn trong code và docs.
- [ ] Có article-level aggregation.
- [ ] Có answer generator với prompt/template chuẩn legal QA.
- [ ] Có citation validator.
- [ ] Có submission schema/exporter/validator.
- [ ] Có script `generate_submission.py`.
- [ ] Có script `validate_submission.py`.
- [ ] Có evaluator Precision / Recall / F2 macro.
- [ ] Có script `evaluate_submission.py`.
- [ ] Có unit test cho metadata, aggregation, submission format, F2 metric.
- [ ] Chạy được sample end-to-end trên 3–5 câu hỏi giả lập hoặc dev sample.
- [ ] Cập nhật `CHANGELOG.md`.
- [ ] Cập nhật bảng trạng thái Task 2 trong file này.

## 2.11. Test bắt buộc cho Task 2

Chạy tối thiểu:

```bash
pytest -q
```

Chạy sample pipeline:

```bash
python scripts/generate_submission.py \
  --input data/raw/sample_questions.json \
  --output data/submissions/sample_results.json \
  --config configs/retrieval.yaml

python scripts/validate_submission.py \
  --input data/submissions/sample_results.json

python scripts/evaluate_submission.py \
  --pred data/submissions/sample_results.json \
  --gold data/processed/sample_gold.json \
  --output data/submissions/sample_eval_report.json
```

Nếu chưa có dữ liệu thật, tạo sample nhỏ trong:

```text
data/raw/sample_questions.json
data/processed/sample_gold.json
```

Sample phải có tối thiểu 3 câu hỏi.

## 2.12. Trạng thái Task 2

| Step | Status | Notes |
|---|---|---|
| PLAN | DONE | Xác định metadata schema, article aggregation, answer generation, evaluation và submission layer |
| IMPLEMENT | DONE | Đã thêm module legal-specific, CLI script, sample data và output artifacts |
| SELF-REVIEW | REVIEWED | Rà lại schema, citation, validator, evaluator và đường dẫn CLI |
| TEST | TESTED_PASS | `pytest -q`, `build_corpus`, `generate_submission`, `validate_submission`, `evaluate_submission` đều pass |
| UPDATE STATUS | DONE | Đã cập nhật bảng trạng thái và changelog |

---

# 3. Checklist review cuối cùng trước khi nộp bài

Trước khi tạo `submission.zip`, bắt buộc kiểm tra:

- [ ] File tên đúng là `results.json`.
- [ ] Zip phẳng, không chứa thư mục con.
- [ ] `results.json` là list JSON hợp lệ.
- [ ] Không thiếu câu hỏi trong test set.
- [ ] Không duplicate `id`.
- [ ] Mỗi answer có ít nhất một `Điều X` nếu có căn cứ.
- [ ] `relevant_docs` đúng format `<mã văn bản>|<tên văn bản>`.
- [ ] `relevant_articles` đúng format `<mã văn bản>|<tên văn bản>|<điều>`.
- [ ] Không có secret/API key trong repo.
- [ ] Không dùng closed LLM/API không hợp lệ với rule cuộc thi.
- [ ] Có log model sử dụng, version model, cách tải model để phục vụ reproducibility.

Lệnh zip:

```bash
cd data/submissions
zip submission.zip results.json
```

Kiểm tra zip:

```bash
unzip -l submission.zip
```

Kết quả hợp lệ phải là:

```text
results.json
```

Không được là:

```text
some_folder/results.json
```

---

# 4. Gợi ý thứ tự commit

```text
commit 1: chore: audit and reorganize project structure
commit 2: chore: move legacy modules out of active pipeline
commit 3: docs: update README and architecture docs
commit 4: feat: add legal metadata schema and article aggregation
commit 5: feat: add legal answer generator and citation validator
commit 6: feat: add submission exporter and validator
commit 7: feat: add article-level F2 evaluator
commit 8: test: add unit and sample end-to-end tests
```

---

# 5. Định nghĩa hoàn thành toàn bộ

Project chỉ được coi là sẵn sàng cho vòng nộp thử khi:

- Task 1 và Task 2 đều có status `TESTED`.
- `pytest -q` pass.
- Sample end-to-end pass.
- `results.json` sample pass validator.
- Có báo cáo `sample_eval_report.json`.
- README đủ hướng dẫn để một thành viên khác trong team setup và chạy lại.
- Không có secret trong repo.
- Không có dependency bắt buộc vào closed LLM API.
