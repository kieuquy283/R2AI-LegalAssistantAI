from __future__ import annotations

from typing import Any, Dict

from .base import BaseQueryRewriter


class NoRewrite(BaseQueryRewriter):
    def rewrite(
        self,
        query: str,
        history_text: str,
    ) -> str:
        return query

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self.get_query_from_state(state)
        _, history_text = self.get_history_from_state(state)
        if history_text:
            state["formatted_history"] = history_text

        state["rewritten_query"] = query
        state["queries"] = [query]
        state["rewrite_applied"] = False
        state["query_rewriting"] = {
            "strategy": "none",
            "input_query": query,
            "rewritten_query": query,
            "queries": [query],
            "should_rewrite": False,
            "rewrite_applied": False,
            "decision_reason": "no_rewrite_baseline",
            "decision_confidence": 1.0,
            "query_type": "baseline",
            "validation_passed": True,
            "validation_errors": [],
            "fallback_reason": None,
            "cache_hit": False,
        }
        return state
