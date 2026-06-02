import pytest

from rag.modules.query_rewriting import (
    LLMQueryRewrite,
    NoRewrite,
    RewriteDecision,
    analyze_query_dependency,
    format_history_for_rewrite,
    has_strong_entity_or_code,
    validate_rewrite,
)
from rag.modules.query_rewriting.utils import is_standalone_definition_query


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if not self.responses:
            raise AssertionError("No fake response configured.")
        return FakeResponse(self.responses.pop(0))


def make_turn(role, content):
    return {"role": role, "content": content}


def test_analyze_query_dependency_empty_query():
    decision = analyze_query_dependency("", has_history=True)
    assert decision.should_rewrite is False
    assert decision.reason == "empty_query"


def test_analyze_query_dependency_no_history():
    decision = analyze_query_dependency("Còn chuyển khẩu thì sao?", has_history=False)
    assert decision.should_rewrite is False
    assert decision.reason == "no_history"


@pytest.mark.parametrize(
    ("query", "expected_reason"),
    [
        ("Còn chuyển khẩu thì sao?", "explicit_follow_up"),
        ("Phạt bao nhiêu?", "short_ambiguous_with_history"),
        ("Trong trường hợp này xử lý như thế nào?", "pronoun_reference"),
    ],
)
def test_analyze_query_dependency_rewrite_cases(query, expected_reason):
    decision = analyze_query_dependency(query, has_history=True)
    assert decision.should_rewrite is True
    assert decision.reason == expected_reason


@pytest.mark.parametrize(
    "query",
    [
        "RAG là gì?",
        "A11 là gì?",
        "Nhập kinh doanh là gì?",
        "FAISS là gì?",
        "HS code là gì?",
    ],
)
def test_standalone_queries_do_not_rewrite(query):
    decision = analyze_query_dependency(query, has_history=True)
    assert decision.should_rewrite is False
    assert decision.reason == "standalone_definition"
    assert is_standalone_definition_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "A11 là gì?",
        "E11 thì sao?",
        "RAG là gì?",
        "FAISS là gì?",
        "HS code là gì?",
        "Theo Điều 12 bị phạt gì?",
        "Nghị định 128 quy định gì?",
    ],
)
def test_has_strong_entity_or_code_positive(query):
    assert has_strong_entity_or_code(query) is True


def test_has_strong_entity_or_code_does_not_flag_first_word_only():
    assert has_strong_entity_or_code("Nhập kinh doanh là gì?") is False


def test_validate_rewrite_rejects_empty_and_answer_style():
    decision = RewriteDecision(True, "explicit_follow_up", 0.9, "follow_up")

    empty_result = validate_rewrite("Phạt bao nhiêu?", "", decision)
    answer_result = validate_rewrite("Phạt bao nhiêu?", "Answer: mức phạt là 10 triệu", decision)
    prefix_result = validate_rewrite("Phạt bao nhiêu?", "Rewritten query: mức phạt là bao nhiêu?", decision)

    assert empty_result.passed is False
    assert "empty_rewrite" in empty_result.errors
    assert answer_result.passed is False
    assert "answer_style_output" in answer_result.errors
    assert prefix_result.passed is False
    assert "answer_style_output" in prefix_result.errors


def test_validate_rewrite_preserves_numbers_and_codes():
    decision = RewriteDecision(False, "standalone_definition", 0.95, "standalone")

    missing_code = validate_rewrite("A11 là gì?", "Loại hình này là gì?", decision)
    missing_number = validate_rewrite("Điều 12 quy định gì?", "Quy định này là gì?", decision)

    assert missing_code.passed is False
    assert any(error.startswith("missing_entity:") for error in missing_code.errors)
    assert missing_number.passed is False
    assert "missing_number:12" in missing_number.errors


def test_validate_rewrite_allows_good_follow_up_rewrite_with_low_overlap():
    decision = RewriteDecision(True, "short_ambiguous_with_history", 0.88, "ambiguous")
    result = validate_rewrite(
        "Phạt bao nhiêu?",
        "Mức phạt khi hàng đã lên chuyền sau khi cắt chì hải quan là bao nhiêu?",
        decision,
    )
    assert result.passed is True


def test_no_rewrite_sets_metadata_and_queries():
    rewriter = NoRewrite()
    state = rewriter.run({"question": "RAG là gì?", "selected_history": [make_turn("user", "Hi")]})

    assert state["rewritten_query"] == "RAG là gì?"
    assert state["queries"] == ["RAG là gì?"]
    assert state["rewrite_applied"] is False
    assert state["query_rewriting"]["strategy"] == "none"


def test_llm_query_rewrite_skips_llm_when_decision_is_no_rewrite(monkeypatch):
    fake_llm = FakeLLM(["should not be used"])
    monkeypatch.setattr("rag.modules.query_rewriting.llm_rewrite.get_llm", lambda **_: fake_llm)

    rewriter = LLMQueryRewrite()
    state = rewriter.run(
        {
            "query": "RAG là gì?",
            "selected_history": [make_turn("user", "RAG dùng để làm gì?")],
        }
    )

    assert fake_llm.calls == 0
    assert state["rewritten_query"] == "RAG là gì?"
    assert state["query_rewriting"]["should_rewrite"] is False
    assert state["query_rewriting"]["decision_reason"] == "standalone_definition"


def test_llm_query_rewrite_uses_cache_without_calling_llm(monkeypatch):
    fake_llm = FakeLLM(["Mức phạt khi hàng đã lên chuyền sau khi cắt chì hải quan là bao nhiêu?"])
    monkeypatch.setattr("rag.modules.query_rewriting.llm_rewrite.get_llm", lambda **_: fake_llm)

    history = [
        make_turn("user", "Hàng đã lên chuyền sau khi cắt chì hải quan thì sao?"),
        make_turn("assistant", "Đây là tình huống rủi ro."),
    ]
    state = {"query": "Phạt bao nhiêu?", "selected_history": history}

    rewriter = LLMQueryRewrite(use_cache=True)
    first_state = rewriter.run(dict(state))
    second_state = rewriter.run(dict(state))

    assert fake_llm.calls == 1
    assert first_state["query_rewriting"]["cache_hit"] is False
    assert second_state["query_rewriting"]["cache_hit"] is True
    assert second_state["rewritten_query"] == first_state["rewritten_query"]


def test_llm_query_rewrite_invalid_output_falls_back(monkeypatch):
    fake_llm = FakeLLM(["Answer: mức phạt là 10 triệu"])
    monkeypatch.setattr("rag.modules.query_rewriting.llm_rewrite.get_llm", lambda **_: fake_llm)

    rewriter = LLMQueryRewrite(use_cache=False)
    state = rewriter.run(
        {
            "current_query": "Phạt bao nhiêu?",
            "selected_history": [
                make_turn("user", "Hàng đã lên chuyền sau khi cắt chì hải quan thì sao?"),
                make_turn("assistant", "Đây là tình huống rủi ro."),
            ],
        }
    )

    assert fake_llm.calls == 1
    assert state["rewritten_query"] == "Phạt bao nhiêu?"
    assert state["rewrite_applied"] is False
    assert state["query_rewriting"]["validation_passed"] is False
    assert state["query_rewriting"]["fallback_reason"] == "invalid_rewrite"


def test_format_history_for_rewrite_uses_clear_roles_and_truncation():
    history = [
        make_turn("human", "Nhập kinh doanh A11 là gì?"),
        make_turn("ai", "A11 là loại hình nhập khẩu để kinh doanh."),
        make_turn("user", "Còn chuyển khẩu thì sao?"),
    ]

    formatted = format_history_for_rewrite(history, max_messages=2)

    assert formatted == (
        "Assistant: A11 là loại hình nhập khẩu để kinh doanh.\n"
        "User: Còn chuyển khẩu thì sao?"
    )
