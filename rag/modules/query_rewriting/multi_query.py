from __future__ import annotations

from typing import Dict, List, Optional

from rag.config.llm import REWRITE_MODEL
from rag.generation.llm_client import get_llm

from .utils import clean_rewritten_query


MULTI_QUERY_PROMPT = """
You generate alternative retrieval queries for a legal RAG system.

Given one standalone query, produce {num_additional_queries} alternative retrieval queries.

Rules:
- Do not answer the question.
- Do not explain anything.
- Keep the original meaning.
- Preserve legal references, numbers, codes, and domain-specific terms.
- Use Vietnamese if the input is Vietnamese.
- Return one query per line.
- Do not number the lines.
- Do not repeat the original query exactly unless needed.

Original Query:
{query}

Alternative Queries:
""".strip()


class MultiQueryGenerator:
    def __init__(
        self,
        model_name: str = REWRITE_MODEL,
        temperature: float = 0.0,
        num_queries: int = 4,
    ) -> None:
        if num_queries <= 0:
            raise ValueError("num_queries must be > 0")
        self.model_name = model_name
        self.temperature = temperature
        self.num_queries = int(num_queries)
        self.llm = get_llm(
            model_name=self.model_name,
            temperature=self.temperature,
        )

    def _parse_queries(
        self,
        raw_output: str,
        original_query: str,
        num_queries: int,
    ) -> List[str]:
        normalized_original = clean_rewritten_query(original_query)
        variants: List[str] = [normalized_original]
        seen = {normalized_original.lower()}

        for raw_line in str(raw_output or "").splitlines():
            cleaned = clean_rewritten_query(raw_line)
            if not cleaned:
                continue
            if cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            variants.append(cleaned)
            if len(variants) >= num_queries:
                break

        return variants[:num_queries]

    def generate(
        self,
        query: str,
        num_queries: Optional[int] = None,
    ) -> List[str]:
        normalized_query = clean_rewritten_query(query)
        if not normalized_query:
            return []

        requested_num_queries = int(num_queries or self.num_queries)
        if requested_num_queries <= 1:
            return [normalized_query]

        prompt = MULTI_QUERY_PROMPT.format(
            query=normalized_query,
            num_additional_queries=requested_num_queries - 1,
        )
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        queries = self._parse_queries(
            raw_output=str(content or ""),
            original_query=normalized_query,
            num_queries=requested_num_queries,
        )
        return queries or [normalized_query]

    def run(self, state: Dict[str, object]) -> Dict[str, object]:
        query = str(state.get("rewritten_query") or state.get("query") or state.get("question") or "").strip()
        generated_queries = self.generate(query=query, num_queries=self.num_queries)
        state["queries"] = generated_queries or ([query] if query else [])
        state["multi_query"] = {
            "strategy": "llm_multi_query",
            "input_query": query,
            "queries": list(state["queries"]),
            "num_queries": len(state["queries"]),
            "model_name": self.model_name,
        }
        return state
