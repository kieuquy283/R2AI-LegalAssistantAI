from __future__ import annotations

import math
import unittest

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


class HistorySelectionTests(unittest.TestCase):
    def test_noise_turns_are_removed(self):
        for content in ("ok", "cảm ơn", "hello"):
            with self.subTest(content=content):
                self.assertFalse(is_meaningful_turn(make_turn("user", content)))

    def test_short_but_meaningful_turns_are_preserved(self):
        samples = (
            "A11 là gì?",
            "E11 thì sao?",
            "Phạt bao nhiêu?",
            "Điều 12?",
            "Còn chuyển khẩu?",
        )

        for content in samples:
            with self.subTest(content=content):
                self.assertTrue(is_meaningful_turn(make_turn("user", content)))

    def test_recency_selector_preserves_order_and_count(self):
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

        self.assertEqual(len(state["selected_history"]), 3)
        self.assertEqual(state["selected_history"], history[3:4] + history[5:7])
        self.assertEqual(state["history_selection"]["strategy"], "recency")
        self.assertEqual(state["history_selection"]["num_selected"], 3)

    def test_base_history_selector_is_backward_compatible_alias(self):
        selector = BaseHistorySelector(top_k=1)
        state = selector.run({"history": [make_turn("user", "Điều 12?")]})
        self.assertEqual(state["selected_history"][0]["content"], "Điều 12?")
        self.assertEqual(state["history_selection"]["strategy"], "recency")

    def test_no_history_selector_sets_empty_selection_and_metadata(self):
        selector = NoHistorySelector()
        state = selector.run({"history": [make_turn("user", "A11 là gì?")]})

        self.assertEqual(state["selected_history"], [])
        self.assertEqual(state["history_selection"]["strategy"], "none")
        self.assertEqual(state["history_selection"]["num_selected"], 0)

    def test_hybrid_selector_handles_empty_history(self):
        selector = HybridHistorySelector(
            embedding_model=FakeEmbeddingModel(),
            top_k=2,
            alpha=3,
            beta=1,
            recent_window=5,
        )

        state = selector.run({"query": "A11 là gì?", "history": []})

        self.assertEqual(state["selected_history"], [])
        self.assertEqual(state["history_selection"]["strategy"], "hybrid")
        self.assertEqual(state["history_selection"]["num_input_history"], 0)
        self.assertEqual(state["history_selection"]["num_selected"], 0)
        self.assertEqual(state["history_selection"]["selected_scores"], [])

    def test_hybrid_selector_handles_empty_query(self):
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

        self.assertEqual(state["selected_history"], [])
        self.assertEqual(state["history_selection"]["num_meaningful_history"], 2)
        self.assertEqual(state["history_selection"]["selected_scores"], [])

    def test_hybrid_selector_adds_metadata_and_preserves_original_order(self):
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

        self.assertTrue(math.isclose(selector.alpha, 0.75))
        self.assertTrue(math.isclose(selector.beta, 0.25))
        self.assertEqual(metadata["strategy"], "hybrid")
        self.assertEqual(metadata["alpha"], 0.75)
        self.assertEqual(metadata["beta"], 0.25)
        self.assertEqual(metadata["recent_window"], 4)
        self.assertEqual(metadata["num_selected"], 2)
        self.assertEqual(
            [turn["content"] for turn in state["selected_history"]],
            [
                "Còn chuyển khẩu?",
                "Chuyển khẩu liên quan đến hàng qua cửa khẩu.",
            ],
        )

        for item in metadata["selected_scores"]:
            self.assertEqual(
                set(item),
                {
                    "original_index",
                    "role",
                    "content",
                    "semantic_score",
                    "recency_score",
                    "final_score",
                },
            )

        self.assertEqual(
            [item["original_index"] for item in metadata["selected_scores"]],
            sorted(item["original_index"] for item in metadata["selected_scores"]),
        )

    def test_semantic_selector_uses_semantic_strategy_metadata(self):
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

        self.assertEqual(state["history_selection"]["strategy"], "semantic")
        self.assertEqual(state["history_selection"]["alpha"], 1.0)
        self.assertEqual(state["history_selection"]["beta"], 0.0)

    def test_hybrid_selector_validates_alpha_beta(self):
        with self.assertRaises(ValueError):
            HybridHistorySelector(
                embedding_model=FakeEmbeddingModel(),
                top_k=1,
                alpha=0,
                beta=0,
                recent_window=5,
            )

    def test_format_history_preserves_order_and_supports_max_chars(self):
        history = [
            make_turn("human", "A11 là gì?"),
            make_turn("ai", "A11 là mã loại hình."),
        ]

        formatted = format_history(history)
        limited = format_history(history, max_chars=20)

        self.assertEqual(formatted, "User: A11 là gì?\nAssistant: A11 là mã loại hình.")
        self.assertEqual(limited, "User: A11 là gì?")


if __name__ == "__main__":
    unittest.main()
