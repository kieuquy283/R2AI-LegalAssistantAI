from __future__ import annotations

from typing import Any, Dict, List

from rag.generation.llm_client import get_llm
from rag.generation.prompt_builder import build_answer_prompt, build_fallback_prompt

INSUFFICIENT_SENTINEL = "__INSUFFICIENT_CONTEXT__"


def generate_answer(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời cuối.
    """
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content.strip()


def answer_with_context_policy(
    question: str,
    rewritten_query: str,
    docs: List[Any],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Chính sách trả lời:
    - Nếu không có docs -> fallback ngay
    - Nếu có docs -> trả lời grounded
    - Nếu grounded trả về sentinel thiếu ngữ cảnh -> fallback có cảnh báo
    """
    if not docs:
        fallback_prompt = build_fallback_prompt(
            question=question,
            history=history,
        )
        fallback_answer = generate_answer(fallback_prompt)

        warning = (
            "⚠️ Cảnh báo: Hệ thống không tìm thấy tài liệu liên quan trong kho tri thức. "
            "Câu trả lời dưới đây được sinh bởi LLM, không dựa trên tài liệu nội bộ."
        )

        return {
            "answer": fallback_answer,
            "mode": "fallback_no_docs",
            "grounded": False,
            "warning": warning,
        }

    grounded_prompt = build_answer_prompt(
        current_question=question,
        rewritten_query=rewritten_query,
        docs=docs,
        history=history,
    )
    grounded_answer = generate_answer(grounded_prompt).strip()

    if grounded_answer == INSUFFICIENT_SENTINEL:
        fallback_prompt = build_fallback_prompt(
            question=question,
            history=history,
        )
        fallback_answer = generate_answer(fallback_prompt)

        warning = (
            "⚠️ Cảnh báo: Có tài liệu được truy xuất nhưng không đủ thông tin để trả lời chắc chắn. "
            "Câu trả lời dưới đây được LLM suy luận thêm và có thể không bám hoàn toàn vào tài liệu trong hệ thống."
        )

        return {
            "answer": fallback_answer,
            "mode": "fallback_insufficient_context",
            "grounded": False,
            "warning": warning,
        }

    return {
        "answer": grounded_answer,
        "mode": "grounded",
        "grounded": True,
        "warning": "",
    }


def stream_answer_with_context_policy(
    question: str,
    rewritten_query: str,
    docs: List[Any],
    history: List[Dict[str, Any]],
):
    """
    Chính sách trả lời dạng luồng (SSE):
    - Xác định mode và prompt trước khi gọi LLM.
    - Trả về dictionary metadata và generator stream các token.
    """
    if not docs:
        fallback_prompt = build_fallback_prompt(
            question=question,
            history=history,
        )
        warning = (
            "⚠️ Cảnh báo: Hệ thống không tìm thấy tài liệu liên quan trong kho tri thức. "
            "Câu trả lời dưới đây được sinh bởi LLM, không dựa trên tài liệu nội bộ."
        )
        metadata = {
            "mode": "fallback_no_docs",
            "grounded": False,
            "warning": warning,
        }
        prompt = fallback_prompt
    else:
        grounded_prompt = build_answer_prompt(
            current_question=question,
            rewritten_query=rewritten_query,
            docs=docs,
            history=history,
        )
        metadata = {
            "mode": "grounded",
            "grounded": True,
            "warning": "",
        }
        prompt = grounded_prompt

    llm = get_llm()
    return metadata, llm.stream(prompt)