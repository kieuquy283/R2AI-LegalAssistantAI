from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from .formatter import format_history_for_rewrite


class BaseQueryRewriter(ABC):
    @abstractmethod
    def rewrite(
        self,
        query: str,
        history_text: str,
    ) -> str:
        pass

    def get_query_from_state(self, state: Dict[str, Any]) -> str:
        return (
            state.get("query")
            or state.get("question")
            or state.get("current_query")
            or ""
        ).strip()

    def get_history_from_state(self, state: Dict[str, Any]) -> Tuple[List[dict], str]:
        selected_history = state.get("selected_history", []) or []
        formatted_history = (state.get("formatted_history") or "").strip()
        if not formatted_history and selected_history:
            formatted_history = format_history_for_rewrite(selected_history)
        return selected_history, formatted_history

    def build_base_metadata(
        self,
        strategy: str,
        query: str,
        rewritten_query: str,
    ) -> Dict[str, Any]:
        rewrite_applied = rewritten_query != query
        return {
            "strategy": strategy,
            "input_query": query,
            "rewritten_query": rewritten_query,
            "queries": [rewritten_query],
            "should_rewrite": rewrite_applied,
            "rewrite_applied": rewrite_applied,
            "decision_reason": "base_rewriter",
            "decision_confidence": 1.0 if rewrite_applied else 0.0,
            "query_type": "unknown",
            "validation_passed": True,
            "validation_errors": [],
            "fallback_reason": None,
            "cache_hit": False,
        }

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self.get_query_from_state(state)
        _, history_text = self.get_history_from_state(state)
        if history_text:
            state["formatted_history"] = history_text

        rewritten_query = self.rewrite(query=query, history_text=history_text) or query
        rewrite_applied = rewritten_query != query

        state["rewritten_query"] = rewritten_query
        state["queries"] = [rewritten_query]
        state["rewrite_applied"] = rewrite_applied
        state["query_rewriting"] = self.build_base_metadata(
            strategy=self.__class__.__name__.lower(),
            query=query,
            rewritten_query=rewritten_query,
        )
        return state

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
