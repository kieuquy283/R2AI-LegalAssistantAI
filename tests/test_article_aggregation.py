from legal_rag.aggregation import ArticleAggregator
from legal_rag.corpus.schema import LegalArticle
from legal_rag.retrieval.keyword import RetrievedChunk


def make_article(article_number: str, score_suffix: str) -> LegalArticle:
    return LegalArticle(
        doc_id="64/2025/QH15",
        doc_title="Luật Ban hành văn bản quy phạm pháp luật",
        doc_full_name="Luật Ban hành văn bản quy phạm pháp luật số 64/2025/QH15",
        article_id=f"64/2025/QH15|Luật Ban hành văn bản quy phạm pháp luật|{article_number}",
        article_number=article_number,
        article_title="Tiêu đề",
        clause_number=None,
        chunk_id=f"chunk_{score_suffix}",
        chunk_text=f"{article_number}. Nội dung",
    )


def test_article_aggregator_groups_duplicate_chunks():
    article = make_article("Điều 4", "a")
    chunks = [
        RetrievedChunk(article=article, score=0.9, dense_score=0.9),
        RetrievedChunk(article=article, score=0.7, dense_score=0.7),
    ]

    selected = ArticleAggregator().select(chunks, query="quy định")

    assert len(selected) == 1
    assert selected[0].article_id == article.article_id
    assert selected[0].score == 0.9


def test_article_aggregator_dynamic_threshold_keeps_close_articles():
    article1 = make_article("Điều 1", "1")
    article2 = make_article("Điều 2", "2")
    chunks = [
        RetrievedChunk(article=article1, score=0.8, dense_score=0.8),
        RetrievedChunk(article=article2, score=0.75, dense_score=0.75),
    ]

    selected = ArticleAggregator(relative_threshold=0.9).select(chunks, query="điều kiện")

    assert len(selected) == 2
