from rag.ingestion.chunkers import split_legal_articles


def test_split_legal_articles():
    text = '''
Điều 1. Quy định chung
Nội dung 1

Điều 2. Quy định tiếp theo
Nội dung 2
'''.strip()
    chunks = split_legal_articles(text)
    assert len(chunks) == 2
    assert chunks[0].startswith("Điều 1")
    assert chunks[1].startswith("Điều 2")
