import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Extract unique documents
print("Extracting unique documents...")
with open("data/embeddings/r2ai_expanded_corpus/expanded_documents.jsonl", "r", encoding="utf-8") as f:
    docs = [json.loads(line) for line in f if line.strip()]

unique_docs = {}
for d in docs:
    doc_id = d["doc_id"]
    if doc_id not in unique_docs:
        unique_docs[doc_id] = {
            "doc_id": doc_id,
            "doc_title": d["doc_title"],
            "doc_number": d.get("doc_number", ""),
            "doc_type": d.get("doc_type", ""),
            "issuer": d.get("issuer", ""),
            "domain": d.get("domain", ""),
            "effect_status": d.get("effect_status", ""),
            "cleaned_text": d.get("cleaned_text", "")
        }

with open("kaggle_data/documents.jsonl", "w", encoding="utf-8") as f:
    for doc_id, doc in unique_docs.items():
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print(f"Saved {len(unique_docs)} unique documents to kaggle_data/documents.jsonl")

# 2. Extract unique articles from chunks
print("Extracting unique articles from chunks...")
with open("data/embeddings/r2ai_expanded_corpus/expanded_chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

unique_articles = {}
for c in chunks:
    article_key = f"{c['doc_id']}_{c.get('article', '')}"
    if article_key not in unique_articles and c.get("article"):
        unique_articles[article_key] = {
            "article_id": article_key,
            "doc_id": c["doc_id"],
            "doc_title": c["doc_title"],
            "doc_number": c.get("doc_number", ""),
            "article": c.get("article", ""),
            "article_title": c.get("article_title", ""),
            "domain": c.get("domain", ""),
            "text": c.get("text", "")
        }

with open("kaggle_data/articles.jsonl", "w", encoding="utf-8") as f:
    for key, article in unique_articles.items():
        f.write(json.dumps(article, ensure_ascii=False) + "\n")

print(f"Saved {len(unique_articles)} unique articles to kaggle_data/articles.jsonl")
