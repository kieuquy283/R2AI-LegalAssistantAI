# Metadata Schema

## LegalArticle

Article-level metadata is standardized in `src/legal_rag/corpus/schema.py`.

```python
class LegalArticle(BaseModel):
    doc_id: str
    doc_title: str
    doc_full_name: str
    article_id: str
    article_number: str
    article_title: Optional[str]
    clause_number: Optional[str]
    chunk_id: str
    chunk_text: str
    source_path: Optional[str]
    effective_date: Optional[str]
    expiry_date: Optional[str]
```

## Required formatting rules

- `doc_ref` must be derived as `<mã văn bản>|<tên văn bản>`
- `article_id` must be `<mã văn bản>|<tên văn bản>|<Điều X>`
- `article_number` is normalized to the `Điều X` form only
- non-article chunks are excluded from article-level submission metadata

## Normalization flow

`scripts/build_corpus.py` reads `data/legal_corpus_chunks.json`, infers document identity from preamble chunks, extracts article headings, and writes normalized JSONL rows to `data/processed/articles.jsonl`.
