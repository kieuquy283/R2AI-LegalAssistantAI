from rag.retrieval.query_rewriter import clean_rewritten_query, is_likely_follow_up


def test_is_likely_follow_up_positive():
    assert is_likely_follow_up("Vậy trường hợp đó thì sao?") is True


def test_is_likely_follow_up_negative():
    assert is_likely_follow_up("Hợp đồng lao động là gì?") is False


def test_clean_rewritten_query():
    raw = "Rewritten query: quyền và nghĩa vụ của người lao động"
    assert clean_rewritten_query(raw) == "quyền và nghĩa vụ của người lao động"
