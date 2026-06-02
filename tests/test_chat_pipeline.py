from rag.schemas.chat import ChatResult


def test_chat_result_schema():
    result = ChatResult(
        answer="ok",
        rewritten_query="q",
        used_rewrite=False,
        mode="grounded",
    )
    assert result.answer == "ok"
    assert result.mode == "grounded"
