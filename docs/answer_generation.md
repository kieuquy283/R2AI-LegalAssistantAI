# Answer Generation

## Goal

Generate Vietnamese legal answers that stay grounded in retrieved evidence and always cite explicit legal articles.

## Output structure

Each answer follows this layout:

1. `Kết luận ngắn`
2. `Căn cứ pháp lý`
3. `Giải thích dễ hiểu`
4. `Gợi ý áp dụng thực tế`
5. `Lưu ý`

## Citation rules

- The answer must mention at least one `Điều X` when `relevant_articles` is not empty.
- Citations must come from selected evidence only.
- If the generated answer misses the legal-basis line, `citation_validator.ensure_citations(...)` prepends one automatically.

## Current implementation

`src/legal_rag/generation/answer_generator.py` builds a deterministic grounded answer from selected articles so local testing and submission generation work without depending on a remote LLM.
