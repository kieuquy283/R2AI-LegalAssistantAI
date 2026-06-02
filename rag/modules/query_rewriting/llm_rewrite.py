from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag.config.llm import REWRITE_MODEL
from rag.generation.llm_client import get_llm

from .base import BaseQueryRewriter
from .prompts import REWRITE_PROMPT
from .rewrite_cache import RewriteCache
from .utils import (
    RewriteDecision,
    RewriteValidationResult,
    analyze_query_dependency,
    clean_rewritten_query,
    validate_rewrite,
)


class LLMQueryRewrite(BaseQueryRewriter):
    def __init__(
        self,
        model_name: str = REWRITE_MODEL,
        temperature: float = 0.0,
        max_rewrite_ratio: float = 2.0,
        max_tokens_multiplier: int = 20,
        use_cache: bool = True,
        cache_size: int = 1000,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_rewrite_ratio = max_rewrite_ratio
        self.max_tokens_multiplier = max_tokens_multiplier
        self.use_cache = use_cache
        self.cache = RewriteCache(max_size=cache_size) if use_cache else None
        self.llm = get_llm(
            model_name=self.model_name,
            temperature=self.temperature,
        )

    def analyze_query_dependency(self, query: str, has_history: bool) -> RewriteDecision:
        return analyze_query_dependency(query, has_history=has_history)

    def should_rewrite(
        self,
        query: str,
        selected_history: List[Dict[str, Any]],
    ) -> bool:
        return self.analyze_query_dependency(query, bool(selected_history)).should_rewrite

    def validate_rewrite(
        self,
        original_query: str,
        rewritten_query: str,
        decision: RewriteDecision,
    ) -> RewriteValidationResult:
        return validate_rewrite(
            original_query=original_query,
            rewritten_query=rewritten_query,
            decision=decision,
            max_rewrite_ratio=self.max_rewrite_ratio,
            max_tokens_multiplier=self.max_tokens_multiplier,
        )

    def _invoke_llm(
        self,
        query: str,
        history_text: str,
    ) -> str:
        prompt = REWRITE_PROMPT.format(history=history_text, query=query)
        response = self.llm.invoke(prompt)
        return clean_rewritten_query(getattr(response, "content", response))

    def rewrite(
        self,
        query: str,
        history_text: str,
    ) -> str:
        return self._invoke_llm(query=query, history_text=history_text)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self.get_query_from_state(state)
        selected_history, history_text = self.get_history_from_state(state)
        state["formatted_history"] = history_text

        decision = self.analyze_query_dependency(query, has_history=bool(selected_history))

        rewritten_query = query
        validation = RewriteValidationResult(True, [])
        fallback_reason: Optional[str] = None
        cache_hit = False

        if decision.should_rewrite:
            cached_rewrite = None
            if self.cache is not None:
                cached_rewrite = self.cache.get(query=query, history_text=history_text)

            if cached_rewrite is not None:
                rewritten_query = cached_rewrite
                cache_hit = True
                validation = self.validate_rewrite(query, rewritten_query, decision)
                if not validation.passed:
                    fallback_reason = "invalid_cached_rewrite"
                    rewritten_query = query
            else:
                llm_rewrite = self.rewrite(query=query, history_text=history_text)
                validation = self.validate_rewrite(query, llm_rewrite, decision)

                if validation.passed:
                    rewritten_query = llm_rewrite
                    if self.cache is not None:
                        self.cache.set(
                            query=query,
                            history_text=history_text,
                            rewritten_query=rewritten_query,
                        )
                else:
                    fallback_reason = "invalid_rewrite"
                    rewritten_query = query

        state["rewritten_query"] = rewritten_query
        state["queries"] = [rewritten_query]
        state["rewrite_applied"] = rewritten_query != query
        state["query_rewriting"] = {
            "strategy": "llm",
            "input_query": query,
            "rewritten_query": rewritten_query,
            "queries": [rewritten_query],
            "should_rewrite": decision.should_rewrite,
            "rewrite_applied": rewritten_query != query,
            "decision_reason": decision.reason,
            "decision_confidence": float(decision.confidence),
            "query_type": decision.query_type,
            "validation_passed": bool(validation.passed),
            "validation_errors": list(validation.errors),
            "fallback_reason": fallback_reason,
            "cache_hit": cache_hit,
            "model_name": self.model_name,
        }
        return state

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name}, "
            f"temperature={self.temperature}, "
            f"use_cache={self.use_cache})"
        )


class MultiQueryRewrite(LLMQueryRewrite):
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = super().run(state)
        primary_query = state["rewritten_query"]
        state["queries"] = [
            primary_query,
            primary_query,
            primary_query,
        ]
        state["query_rewriting"]["strategy"] = "multi_query_placeholder"
        state["query_rewriting"]["queries"] = list(state["queries"])
        return state
