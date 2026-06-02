from __future__ import annotations

from typing import Any, Dict, List, Tuple

from rag.config.llm import REWRITE_MODEL
from rag.generation.llm_client import get_llm

from .formatter import format_history_for_rewrite
from .utils import clean_rewritten_query


HYDE_PROMPT = """
For Vietnamese legal retrieval, generate a concise hypothetical legal passage that likely contains the terms and legal concepts needed to retrieve the correct law article.

Rules:
- Return only the hypothetical retrieval text.
- Do not include explanations.
- Do not include citations.
- Use Vietnamese if the input is Vietnamese.
- Keep legal terms, article references, numbers, and domain concepts precise.

Conversation History:
{history}

Standalone Query:
{query}

Hypothetical Retrieval Text:
""".strip()


class HyDEQueryGenerator:
    def __init__(
        self,
        model_name: str = REWRITE_MODEL,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.llm = get_llm(
            model_name=self.model_name,
            temperature=self.temperature,
        )

    def generate(
        self,
        rewritten_query: str,
        selected_history: List[Dict[str, Any]] | None = None,
    ) -> str:
        normalized_query = clean_rewritten_query(rewritten_query)
        if not normalized_query:
            return ""

        history_text = ""
        if selected_history:
            history_text = format_history_for_rewrite(selected_history)

        prompt = HYDE_PROMPT.format(
            history=history_text or "(không có)",
            query=normalized_query,
        )
        response = self.llm.invoke(prompt)
        return clean_rewritten_query(getattr(response, "content", response))

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        rewritten_query = str(
            state.get("rewritten_query")
            or state.get("query")
            or state.get("question")
            or ""
        ).strip()
        selected_history = list(state.get("selected_history", []) or [])
        hyde_query, hyde_used, hyde_failed = generate_hyde_query(
            rewritten_query=rewritten_query,
            selected_history=selected_history,
            mode="llm",
            generator=self,
        )
        state["hyde_query"] = hyde_query
        state["hyde_used"] = hyde_used
        state["hyde_failed"] = hyde_failed
        state["hyde"] = {
            "strategy": "llm_hyde",
            "input_query": rewritten_query,
            "hyde_query": hyde_query,
            "hyde_used": hyde_used,
            "hyde_failed": hyde_failed,
            "model_name": self.model_name,
        }
        return state


def generate_hyde_query(
    rewritten_query: str,
    selected_history: List[Dict[str, Any]] | None = None,
    mode: str = "llm",
    generator: HyDEQueryGenerator | None = None,
) -> Tuple[str, bool, bool]:
    normalized_query = clean_rewritten_query(rewritten_query)
    if mode == "none":
        return normalized_query, False, False

    if not normalized_query:
        return "", False, True

    try:
        active_generator = generator or HyDEQueryGenerator()
        hyde_query = active_generator.generate(
            rewritten_query=normalized_query,
            selected_history=selected_history,
        )
        if not hyde_query:
            return normalized_query, False, True
        return hyde_query, True, False
    except Exception:
        return normalized_query, False, True
