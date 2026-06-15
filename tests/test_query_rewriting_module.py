from __future__ import annotations

import unittest
from unittest.mock import patch

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


class QueryRewritingTests(unittest.TestCase):
    def test_analyze_query_dependency_empty_query(self):
        decision = analyze_query_dependency("", has_history=True)
        self.assertFalse(decision.should_rewrite)
        self.assertEqual(decision.reason, "empty_query")

    def test_analyze_query_dependency_no_history(self):
        decision = analyze_query_dependency("Còn chuyển khẩu thì sao?", has_history=False)
        self.assertFalse(decision.should_rewrite)
        self.assertEqual(decision.reason, "no_history")

    def test_analyze_query_dependency_rewrite_cases(self):
        cases = [
            ("Còn chuyển khẩu thì sao?", "explicit_follow_up"),
            ("Phạt bao nhiêu?", "short_ambiguous_with_history"),
            ("Trong trường hợp này xử lý như thế nào?", "pronoun_reference"),
        ]
        for query, expected_reason in cases:
            with self.subTest(query=query):
                decision = analyze_query_dependency(query, has_history=True)
                self.assertTrue(decision.should_rewrite)
                self.assertEqual(decision.reason, expected_reason)

    def test_standalone_queries_do_not_rewrite(self):
        for query in [
            "RAG là gì?",
            "A11 là gì?",
            "Nhập kinh doanh là gì?",
            "FAISS là gì?",
            "HS code là gì?",
        ]:
            with self.subTest(query=query):
                decision = analyze_query_dependency(query, has_history=True)
                self.assertFalse(decision.should_rewrite)
                self.assertEqual(decision.reason, "standalone_definition")
                self.assertTrue(is_standalone_definition_query(query))

    def test_has_strong_entity_or_code_positive(self):
        for query in [
            "A11 là gì?",
            "E11 thì sao?",
            "RAG là gì?",
            "FAISS là gì?",
            "HS code là gì?",
            "Theo Điều 12 bị phạt gì?",
            "Nghị định 128 quy định gì?",
        ]:
            with self.subTest(query=query):
                self.assertTrue(has_strong_entity_or_code(query))

    def test_has_strong_entity_or_code_does_not_flag_first_word_only(self):
        self.assertFalse(has_strong_entity_or_code("Nhập kinh doanh là gì?"))

    def test_validate_rewrite_rejects_empty_and_answer_style(self):
        decision = RewriteDecision(True, "explicit_follow_up", 0.9, "follow_up")

        empty_result = validate_rewrite("Phạt bao nhiêu?", "", decision)
        answer_result = validate_rewrite("Phạt bao nhiêu?", "Answer: mức phạt là 10 triệu", decision)
        prefix_result = validate_rewrite("Phạt bao nhiêu?", "Rewritten query: mức phạt là bao nhiêu?", decision)

        self.assertFalse(empty_result.passed)
        self.assertIn("empty_rewrite", empty_result.errors)
        self.assertFalse(answer_result.passed)
        self.assertIn("answer_style_output", answer_result.errors)
        self.assertFalse(prefix_result.passed)
        self.assertIn("answer_style_output", prefix_result.errors)

    def test_validate_rewrite_preserves_numbers_and_codes(self):
        decision = RewriteDecision(False, "standalone_definition", 0.95, "standalone")

        missing_code = validate_rewrite("A11 là gì?", "Loại hình này là gì?", decision)
        missing_number = validate_rewrite("Điều 12 quy định gì?", "Quy định này là gì?", decision)

        self.assertFalse(missing_code.passed)
        self.assertTrue(any(error.startswith("missing_entity:") for error in missing_code.errors))
        self.assertFalse(missing_number.passed)
        self.assertIn("missing_number:12", missing_number.errors)

    def test_validate_rewrite_allows_good_follow_up_rewrite_with_low_overlap(self):
        decision = RewriteDecision(True, "short_ambiguous_with_history", 0.88, "ambiguous")
        result = validate_rewrite(
            "Phạt bao nhiêu?",
            "Mức phạt khi hàng đã lên chuyến sau khi cắt chỉ hải quan là bao nhiêu?",
            decision,
        )
        self.assertTrue(result.passed)

    def test_no_rewrite_sets_metadata_and_queries(self):
        rewriter = NoRewrite()
        state = rewriter.run({"question": "RAG là gì?", "selected_history": [make_turn("user", "Hi")]})

        self.assertEqual(state["rewritten_query"], "RAG là gì?")
        self.assertEqual(state["queries"], ["RAG là gì?"])
        self.assertFalse(state["rewrite_applied"])
        self.assertEqual(state["query_rewriting"]["strategy"], "none")

    def test_llm_query_rewrite_skips_llm_when_decision_is_no_rewrite(self):
        fake_llm = FakeLLM(["should not be used"])
        with patch("rag.modules.query_rewriting.llm_rewrite.get_llm", lambda **_: fake_llm):
            rewriter = LLMQueryRewrite()
            state = rewriter.run(
                {
                    "query": "RAG là gì?",
                    "selected_history": [make_turn("user", "RAG dùng để làm gì?")],
                }
            )

        self.assertEqual(fake_llm.calls, 0)
        self.assertEqual(state["rewritten_query"], "RAG là gì?")
        self.assertFalse(state["query_rewriting"]["should_rewrite"])
        self.assertEqual(state["query_rewriting"]["decision_reason"], "standalone_definition")

    def test_llm_query_rewrite_uses_cache_without_calling_llm(self):
        fake_llm = FakeLLM(["Mức phạt khi hàng đã lên chuyến sau khi cắt chỉ hải quan là bao nhiêu?"])
        with patch("rag.modules.query_rewriting.llm_rewrite.get_llm", lambda **_: fake_llm):
            history = [
                make_turn("user", "Hàng đã lên chuyến sau khi cắt chỉ hải quan thì sao?"),
                make_turn("assistant", "Đây là tình huống rủi ro."),
            ]
            state = {"query": "Phạt bao nhiêu?", "selected_history": history}

            rewriter = LLMQueryRewrite(use_cache=True)
            first_state = rewriter.run(dict(state))
            second_state = rewriter.run(dict(state))

        self.assertEqual(fake_llm.calls, 1)
        self.assertFalse(first_state["query_rewriting"]["cache_hit"])
        self.assertTrue(second_state["query_rewriting"]["cache_hit"])
        self.assertEqual(second_state["rewritten_query"], first_state["rewritten_query"])

    def test_llm_query_rewrite_invalid_output_falls_back(self):
        fake_llm = FakeLLM(["Answer: mức phạt là 10 triệu"])
        with patch("rag.modules.query_rewriting.llm_rewrite.get_llm", lambda **_: fake_llm):
            rewriter = LLMQueryRewrite(use_cache=False)
            state = rewriter.run(
                {
                    "current_query": "Phạt bao nhiêu?",
                    "selected_history": [
                        make_turn("user", "Hàng đã lên chuyến sau khi cắt chỉ hải quan thì sao?"),
                        make_turn("assistant", "Đây là tình huống rủi ro."),
                    ],
                }
            )

        self.assertEqual(fake_llm.calls, 1)
        self.assertEqual(state["rewritten_query"], "Phạt bao nhiêu?")
        self.assertFalse(state["rewrite_applied"])
        self.assertFalse(state["query_rewriting"]["validation_passed"])
        self.assertEqual(state["query_rewriting"]["fallback_reason"], "invalid_rewrite")

    def test_format_history_for_rewrite_uses_clear_roles_and_truncation(self):
        history = [
            make_turn("human", "Nhập kinh doanh A11 là gì?"),
            make_turn("ai", "A11 là loại hình nhập khẩu để kinh doanh."),
            make_turn("user", "Còn chuyển khẩu thì sao?"),
        ]

        formatted = format_history_for_rewrite(history, max_messages=2)

        self.assertEqual(
            formatted,
            (
                "Assistant: A11 là loại hình nhập khẩu để kinh doanh.\n"
                "User: Còn chuyển khẩu thì sao?"
            ),
        )


if __name__ == "__main__":
    unittest.main()
