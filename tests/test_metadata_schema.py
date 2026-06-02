from legal_rag.corpus.normalize import normalize_corpus_items
from legal_rag.corpus.schema import LegalArticle


def test_legal_article_schema_and_doc_ref():
    article = LegalArticle(
        doc_id="64/2025/QH15",
        doc_title="Luật Ban hành văn bản quy phạm pháp luật",
        doc_full_name="Luật Ban hành văn bản quy phạm pháp luật số 64/2025/QH15",
        article_id="64/2025/QH15|Luật Ban hành văn bản quy phạm pháp luật|Điều 4",
        article_number="Điều 4",
        article_title="Hệ thống văn bản quy phạm pháp luật",
        clause_number=None,
        chunk_id="64_2025_qh15_dieu_4",
        chunk_text="Điều 4. Hệ thống văn bản quy phạm pháp luật",
    )

    assert article.doc_ref == "64/2025/QH15|Luật Ban hành văn bản quy phạm pháp luật"
    assert article.article_number == "Điều 4"


def test_normalize_corpus_items_extracts_article_metadata():
    items = [
        {
            "doc_id": "64/2025/QH15",
            "chunk_id": "intro",
            "content": "LUẬT\nBAN HÀNH VĂN BẢN QUY PHẠM PHÁP LUẬT\nLuật Ban hành văn bản quy phạm pháp luật số 64/2025/QH15",
        },
        {
            "doc_id": "64/2025/QH15",
            "chunk_id": "dieu_1",
            "content": "Điều 1. Phạm vi điều chỉnh\nLuật này quy định về xây dựng, ban hành văn bản quy phạm pháp luật.",
        },
    ]

    articles = normalize_corpus_items(items)

    assert len(articles) == 1
    assert articles[0].article_id == "64/2025/QH15|Luật Ban hành văn bản quy phạm pháp luật|Điều 1"
