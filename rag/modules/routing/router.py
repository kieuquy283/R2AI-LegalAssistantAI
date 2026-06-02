from __future__ import annotations

from .query_analyzer import analyze_query
from .schemas import PipelineLevel, RetrievalConfidence, RouteDecision


class AdaptiveRouter:
    def __init__(self) -> None:
        self._thresholds = {
            PipelineLevel.SIMPLE_DENSE: 0.68,
            PipelineLevel.REWRITE_DENSE: 0.72,
            PipelineLevel.HYBRID: 0.78,
            PipelineLevel.HYBRID_RERANK: 0.82,
            PipelineLevel.FULL_OPTIMAL: 0.86,
            PipelineLevel.HYDE: 1.01,
        }

    def build_route_decision(
        self,
        level: PipelineLevel,
        *,
        analyzer_signals: dict | None = None,
        reasons: list[str] | None = None,
    ) -> RouteDecision:
        cost_map = {
            PipelineLevel.SIMPLE_DENSE: "cheap",
            PipelineLevel.REWRITE_DENSE: "low",
            PipelineLevel.HYBRID: "medium",
            PipelineLevel.HYBRID_RERANK: "medium_high",
            PipelineLevel.FULL_OPTIMAL: "high",
            PipelineLevel.HYDE: "very_high",
        }
        return RouteDecision(
            level=level,
            analyzer_signals=dict(analyzer_signals or {}),
            reasons=list(reasons or []),
            estimated_cost=cost_map[level],
        )

    def choose_initial_route(self, question: str, history: list[dict] | None) -> RouteDecision:
        signals = analyze_query(question=question, history=history or [])
        reasons: list[str] = []

        if signals["is_very_complex"]:
            reasons.append("very_complex_question")
            return self.build_route_decision(
                PipelineLevel.FULL_OPTIMAL,
                analyzer_signals=signals,
                reasons=reasons,
            )

        if signals["has_followup_markers"] or signals["needs_history"]:
            reasons.append("follow_up_or_history_dependency")
            return self.build_route_decision(
                PipelineLevel.REWRITE_DENSE,
                analyzer_signals=signals,
                reasons=reasons,
            )

        if signals["is_complex"]:
            reasons.append("complex_question")
            return self.build_route_decision(
                PipelineLevel.HYBRID,
                analyzer_signals=signals,
                reasons=reasons,
            )

        if signals["is_standalone"] and signals["has_legal_keywords"] and signals["question_length"] <= 10:
            reasons.append("short_standalone_legal_query")
            return self.build_route_decision(
                PipelineLevel.SIMPLE_DENSE,
                analyzer_signals=signals,
                reasons=reasons,
            )

        if signals["is_abstract"]:
            reasons.append("abstract_question_start_low")
            return self.build_route_decision(
                PipelineLevel.SIMPLE_DENSE,
                analyzer_signals=signals,
                reasons=reasons,
            )

        reasons.append("default_progressive_start")
        return self.build_route_decision(
            PipelineLevel.SIMPLE_DENSE,
            analyzer_signals=signals,
            reasons=reasons,
        )

    def compute_next_level(self, current_level: PipelineLevel) -> PipelineLevel | None:
        ordered_levels = [
            PipelineLevel.SIMPLE_DENSE,
            PipelineLevel.REWRITE_DENSE,
            PipelineLevel.HYBRID,
            PipelineLevel.HYBRID_RERANK,
            PipelineLevel.FULL_OPTIMAL,
            PipelineLevel.HYDE,
        ]
        try:
            current_index = ordered_levels.index(current_level)
        except ValueError:
            return None
        next_index = current_index + 1
        if next_index >= len(ordered_levels):
            return None
        return ordered_levels[next_index]

    def should_escalate(
        self,
        confidence: RetrievalConfidence,
        current_level: PipelineLevel,
    ) -> bool:
        if current_level == PipelineLevel.HYDE:
            return False
        if confidence.doc_count == 0:
            return True
        if confidence.unique_cid_count == 0:
            return True
        if "low_cid_diversity" in confidence.reasons and current_level in {
            PipelineLevel.SIMPLE_DENSE,
            PipelineLevel.REWRITE_DENSE,
            PipelineLevel.HYBRID,
        }:
            return True
        threshold = self._thresholds[current_level]
        return confidence.score < threshold

