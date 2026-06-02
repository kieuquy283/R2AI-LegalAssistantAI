from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .utils import (
    annotate_history,
    build_selection_metadata,
    compute_recency_score,
)


AnnotatedHistory = List[Tuple[int, Dict[str, Any]]]


class NoHistorySelector:
    strategy = "none"

    def __init__(self, top_k: int = 0):
        self.top_k = int(top_k)

    def select(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return []

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        history = state.get("history", []) or []
        state["selected_history"] = []
        state["history_selection"] = build_selection_metadata(
            strategy=self.strategy,
            top_k=self.top_k,
            alpha=None,
            beta=None,
            recent_window=None,
            num_input_history=len(history),
            num_meaningful_history=len(annotate_history(history)),
            num_selected=0,
            selected_scores=[],
        )
        return state

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(top_k={self.top_k})"


class RecencyHistorySelector:
    strategy = "recency"

    def __init__(self, top_k: int = 3):
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        self.top_k = int(top_k)

    def get_meaningful_history(
        self,
        history: List[Dict[str, Any]],
    ) -> AnnotatedHistory:
        return annotate_history(history)

    def score_history(
        self,
        history: List[Dict[str, Any]],
    ) -> Tuple[AnnotatedHistory, List[Dict[str, Any]]]:
        meaningful_history = self.get_meaningful_history(history)
        total_turns = len(meaningful_history)

        selected = meaningful_history[-self.top_k :]
        selected_indices = {original_index for original_index, _ in selected}
        scored_selection: List[Dict[str, Any]] = []

        for recency_index, (original_index, turn) in enumerate(meaningful_history):
            if original_index not in selected_indices:
                continue

            recency_score = compute_recency_score(recency_index, total_turns)
            scored_selection.append(
                {
                    "original_index": int(original_index),
                    "role": str(turn.get("role", "")),
                    "content": str(turn.get("content", "")),
                    "semantic_score": 0.0,
                    "recency_score": float(recency_score),
                    "final_score": float(recency_score),
                }
            )

        scored_selection.sort(key=lambda item: item["original_index"])
        return meaningful_history, scored_selection

    def select(
        self,
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        meaningful_history = self.get_meaningful_history(history)
        selected = meaningful_history[-self.top_k :]
        return [turn for _, turn in selected]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        history = state.get("history", []) or []
        meaningful_history, scored_selection = self.score_history(history)
        selected_history = [history[item["original_index"]] for item in scored_selection]

        state["selected_history"] = selected_history
        state["history_selection"] = build_selection_metadata(
            strategy=self.strategy,
            top_k=self.top_k,
            alpha=0.0,
            beta=1.0,
            recent_window=None,
            num_input_history=len(history),
            num_meaningful_history=len(meaningful_history),
            num_selected=len(selected_history),
            selected_scores=scored_selection,
        )
        return state

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(top_k={self.top_k})"


class BaseHistorySelector(RecencyHistorySelector):
    pass
