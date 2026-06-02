from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .base import RecencyHistorySelector
from .utils import (
    build_selection_metadata,
    compute_recency_score,
    cosine_similarity,
    extract_query_from_state,
    get_turn_content,
)


ScoredTurn = Dict[str, Any]


class HybridHistorySelector(RecencyHistorySelector):
    strategy = "hybrid"

    def __init__(
        self,
        embedding_model: Any,
        top_k: int = 3,
        alpha: float = 0.8,
        beta: float = 0.2,
        recent_window: int = 20,
    ):
        super().__init__(top_k=top_k)

        if recent_window <= 0:
            raise ValueError("recent_window must be > 0")
        if alpha + beta <= 0:
            raise ValueError("alpha + beta must be > 0")

        total = alpha + beta
        self.embedding_model = embedding_model
        self.alpha = float(alpha) / float(total)
        self.beta = float(beta) / float(total)
        self.recent_window = int(recent_window)

    def _embed_history_contents(self, contents: List[str]) -> List[Sequence[float]]:
        embed_documents = getattr(self.embedding_model, "embed_documents", None)
        if callable(embed_documents):
            return list(embed_documents(contents))
        return [self.embedding_model.embed_query(content) for content in contents]

    def rank_history(
        self,
        query: str,
        history: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[ScoredTurn]]:
        meaningful_history = self.get_meaningful_history(history)
        recent_history = meaningful_history[-self.recent_window :]

        if not recent_history or not query.strip():
            return meaningful_history, []

        query_embedding = self.embedding_model.embed_query(query)
        contents = [get_turn_content(turn) for _, turn in recent_history]
        history_embeddings = self._embed_history_contents(contents)
        total_turns = len(recent_history)

        scored_history: List[ScoredTurn] = []

        for recency_index, ((original_index, turn), turn_embedding) in enumerate(
            zip(recent_history, history_embeddings)
        ):
            semantic_score = cosine_similarity(query_embedding, turn_embedding)
            recency_score = compute_recency_score(recency_index, total_turns)
            final_score = (self.alpha * semantic_score) + (self.beta * recency_score)

            scored_history.append(
                {
                    "original_index": int(original_index),
                    "role": str(turn.get("role", "")),
                    "content": str(turn.get("content", "")),
                    "semantic_score": float(semantic_score),
                    "recency_score": float(recency_score),
                    "final_score": float(final_score),
                    "turn": turn,
                }
            )

        scored_history.sort(key=lambda item: item["final_score"], reverse=True)
        return meaningful_history, scored_history

    def score_history(
        self,
        query: str,
        history: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[ScoredTurn]]:
        meaningful_history, ranked_history = self.rank_history(query, history)
        selected = ranked_history[: self.top_k]
        selected.sort(key=lambda item: item["original_index"])

        selected_scores: List[ScoredTurn] = []
        for item in selected:
            selected_scores.append(
                {
                    "original_index": int(item["original_index"]),
                    "role": str(item["role"]),
                    "content": str(item["content"]),
                    "semantic_score": float(item["semantic_score"]),
                    "recency_score": float(item["recency_score"]),
                    "final_score": float(item["final_score"]),
                }
            )

        return meaningful_history, selected_scores

    def select(
        self,
        query: str,
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        _, selected_scores = self.score_history(query, history)
        return [history[item["original_index"]] for item in selected_scores]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        history = state.get("history", []) or []
        query = extract_query_from_state(state)
        meaningful_history, selected_scores = self.score_history(query, history)
        selected_history = [history[item["original_index"]] for item in selected_scores]

        state["selected_history"] = selected_history
        state["history_selection"] = build_selection_metadata(
            strategy=self.strategy,
            top_k=self.top_k,
            alpha=self.alpha,
            beta=self.beta,
            recent_window=self.recent_window,
            num_input_history=len(history),
            num_meaningful_history=len(meaningful_history),
            num_selected=len(selected_history),
            selected_scores=selected_scores,
        )
        return state

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"top_k={self.top_k}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"recent_window={self.recent_window})"
        )


class SemanticHistorySelector(HybridHistorySelector):
    strategy = "semantic"

    def __init__(
        self,
        embedding_model: Any,
        top_k: int = 3,
        recent_window: int = 20,
    ):
        super().__init__(
            embedding_model=embedding_model,
            top_k=top_k,
            alpha=1.0,
            beta=0.0,
            recent_window=recent_window,
        )
