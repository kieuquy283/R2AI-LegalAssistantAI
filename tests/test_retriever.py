from rag.retrieval.ranking import build_top_files


class DummyDoc:
    def __init__(self, source_file, score):
        self.metadata = {"source_file": source_file, "raw_score": score}


def test_build_top_files():
    docs = [
        DummyDoc("a.pdf", 0.1),
        DummyDoc("a.pdf", 0.2),
        DummyDoc("b.pdf", 0.05),
    ]
    results = build_top_files(docs, top_k_files=2)
    assert results[0]["source_file"] == "b.pdf"
    assert results[1]["source_file"] == "a.pdf"
