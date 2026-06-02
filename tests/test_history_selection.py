import math

import pytest

from rag.modules.history_selection import (
    BaseHistorySelector,
    HybridHistorySelector,
    NoHistorySelector,
    RecencyHistorySelector,
    SemanticHistorySelector,
    format_history,
    is_meaningful_turn,
)


class FakeEmbeddingModel:
    def embed_query(self, text):
        text = text or ""
        return [
            float("hải quan" in text.lower() or "chuyển khẩu" in text.lower()),
            float("a11" in text.lower() or "e11" in text.lower()),
            float(len(text)),
        ]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def make_turn(role, content):
    return {"role": role, "content": content}


def test_noise_turns_are_removed():
    for content in ("ok", "cảm ơn", "hello"):
        assert is_meaningful_turn(make_turn("user", content)) is False


def test_short_but_meaningful_turns_are_preserved():
    samples = (
        "A11 là gì?",
        "E11 thì sao?",
        "Phạt bao nhiêu?",
        "Điều 12?",
        "Còn chuyển khẩu?",
    )

    for content in samples:
        assert is_meaningful_turn(make_turn("user", content)) is True


def test_recency_selector_preserves_order_and_count():
    history = [
        make_turn("user", "hello"),
        make_turn("assistant", "Xin chào"),
        make_turn("user", "Tờ khai nhập kinh doanh cần gì?"),
        make_turn("assistant", "Cần bộ hồ sơ hải quan."),
        make_turn("user", "ok"),
        make_turn("user", "Còn chuyển khẩu?"),
        make_turn("assistant", "Chuyển khẩu cần kiểm tra cửa khẩu."),
    ]

    selector = RecencyHistorySelector(top_k=3)
    state = selector.run({"history": history})

    assert len(state["selected_history"]) == 3
    assert state["selected_history"] == history[3:4] + history[5:7]
    assert state["history_selection"]["strategy"] == "recency"
    assert state["history_selection"]["num_selected"] == 3


def test_base_history_selector_is_backward_compatible_alias():
    selector = BaseHistorySelector(top_k=1)
    state = selector.run({"history": [make_turn("user", "Điều 12?")]})
    assert state["selected_history"][0]["content"] == "Điều 12?"
    assert state["history_selection"]["strategy"] == "recency"


def test_no_history_selector_sets_empty_selection_and_metadata():
    selector = NoHistorySelector()
    state = selector.run({"history": [make_turn("user", "A11 là gì?")]})

    assert state["selected_history"] == []
    assert state["history_selection"]["strategy"] == "none"
    assert state["history_selection"]["num_selected"] == 0


def test_hybrid_selector_handles_empty_history():
    selector = HybridHistorySelector(
        embedding_model=FakeEmbeddingModel(),
        top_k=2,
        alpha=3,
        beta=1,
        recent_window=5,
    )

    state = selector.run({"query": "A11 là gì?", "history": []})

    assert state["selected_history"] == []
    assert state["history_selection"]["strategy"] == "hybrid"
    assert state["history_selection"]["num_input_history"] == 0
    assert state["history_selection"]["num_selected"] == 0
    assert state["history_selection"]["selected_scores"] == []


def test_hybrid_selector_handles_empty_query():
    history = [
        make_turn("user", "Tờ khai hải quan là gì?"),
        make_turn("assistant", "Đây là chứng từ khai báo."),
    ]
    selector = HybridHistorySelector(
        embedding_model=FakeEmbeddingModel(),
        top_k=2,
        alpha=1,
        beta=1,
        recent_window=5,
    )

    state = selector.run({"history": history, "query": ""})

    assert state["selected_history"] == []
    assert state["history_selection"]["num_meaningful_history"] == 2
    assert state["history_selection"]["selected_scores"] == []


def test_hybrid_selector_adds_metadata_and_preserves_original_order():
    history = [
        make_turn("user", "A11 là gì?"),
        make_turn("assistant", "A11 là mã loại hình nhập kinh doanh."),
        make_turn("user", "ok"),
        make_turn("user", "Còn chuyển khẩu?"),
        make_turn("assistant", "Chuyển khẩu liên quan đến hàng qua cửa khẩu."),
    ]
    selector = HybridHistorySelector(
        embedding_model=FakeEmbeddingModel(),
        top_k=2,
        alpha=3,
        beta=1,
        recent_window=4,
    )

    state = selector.run({"question": "chuyển khẩu hải quan", "history": history})
    metadata = state["history_selection"]

    assert math.isclose(selector.alpha, 0.75)
    assert math.isclose(selector.beta, 0.25)
    assert metadata["strategy"] == "hybrid"
    assert metadata["alpha"] == pytest.approx(0.75)
    assert metadata["beta"] == pytest.approx(0.25)
    assert metadata["recent_window"] == 4
    assert metadata["num_selected"] == 2
    assert [turn["content"] for turn in state["selected_history"]] == [
        "Còn chuyển khẩu?",
        "Chuyển khẩu liên quan đến hàng qua cửa khẩu.",
    ]

    for item in metadata["selected_scores"]:
        assert set(item) == {
            "original_index",
            "role",
            "content",
            "semantic_score",
            "recency_score",
            "final_score",
        }

    assert [item["original_index"] for item in metadata["selected_scores"]] == sorted(
        item["original_index"] for item in metadata["selected_scores"]
    )


def test_semantic_selector_uses_semantic_strategy_metadata():
    history = [
        make_turn("user", "E11 thì sao?"),
        make_turn("assistant", "E11 là loại hình xuất khẩu."),
    ]
    selector = SemanticHistorySelector(
        embedding_model=FakeEmbeddingModel(),
        top_k=1,
        recent_window=5,
    )

    state = selector.run({"rewritten_query": "E11", "history": history})

    assert state["history_selection"]["strategy"] == "semantic"
    assert state["history_selection"]["alpha"] == 1.0
    assert state["history_selection"]["beta"] == 0.0


def test_hybrid_selector_validates_alpha_beta():
    with pytest.raises(ValueError):
        HybridHistorySelector(
            embedding_model=FakeEmbeddingModel(),
            top_k=1,
            alpha=0,
            beta=0,
            recent_window=5,
        )


def test_format_history_preserves_order_and_supports_max_chars():
    history = [
        make_turn("human", "A11 là gì?"),
        make_turn("ai", "A11 là mã loại hình."),
    ]

    formatted = format_history(history)
    limited = format_history(history, max_chars=20)

    assert formatted == "User: A11 là gì?\nAssistant: A11 là mã loại hình."
    assert limited == "User: A11 là gì?"
