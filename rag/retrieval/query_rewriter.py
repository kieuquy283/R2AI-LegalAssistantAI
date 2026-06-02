"""
Legacy compatibility module.

This file is kept to avoid breaking older pipeline/API/evaluation code.
New code should use rag.modules.query_rewriting.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rag.config.llm import REWRITE_MODEL
from rag.config.retrieval import HISTORY_TURNS
from rag.generation.llm_client import get_llm
from rag.modules.query_rewriting import clean_rewritten_query, is_likely_follow_up
from rag.modules.query_rewriting.formatter import format_history_for_rewrite as modular_format_history_for_rewrite


def format_history_for_rewrite(
    history: List[Dict[str, Any]],
    max_turns: int = HISTORY_TURNS,
) -> str:
    return modular_format_history_for_rewrite(
        history=history,
        max_messages=max_turns * 2,
    )


def rewrite_query(current_question: str, history: List[Dict[str, Any]]) -> str:
    question = current_question.strip()
    if not question:
        raise ValueError("current_question rỗng, không thể rewrite.")

    if not history:
        return question

    if not is_likely_follow_up(question):
        return question

    history_text = format_history_for_rewrite(history, max_turns=HISTORY_TURNS)

    prompt = f"""
You are a query rewriter for a multi-turn RAG system.

Your job is to rewrite the user's latest question into ONE short standalone search query.

Important rules:
- Preserve the meaning exactly.
- Only add missing context from conversation history when necessary.
- Do not answer the question.
- Do not explain anything.
- Do not add extra details not present in the conversation.
- Keep the rewritten query natural, concise, and retrieval-friendly.
- Output only the rewritten query, with no label or commentary.
- If the current question is already standalone enough for retrieval, return it unchanged.

Conversation history:
{history_text}

Current user question:
{question}

Standalone search query:
""".strip()

    llm = get_llm(model_name=REWRITE_MODEL, temperature=0.0)
    response = llm.invoke(prompt)
    rewritten_query = clean_rewritten_query(response.content)

    if not rewritten_query:
        return question

    if len(rewritten_query.split()) > max(20, len(question.split()) * 2):
        return question

    return rewritten_query
