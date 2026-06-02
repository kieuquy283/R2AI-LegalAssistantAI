from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from rag.config.llm import (
    DASHSCOPE_API_KEY,
    QWEN_BASE_URL,
    CHAT_MODEL,
    TEMPERATURE,
    validate_llm_config,
)


def get_llm(
    model: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """
    Khởi tạo chat model.

    Tương thích cả 2 kiểu gọi:
    - get_llm(model="qwen-plus")
    - get_llm(model_name="qwen-plus")
    """
    selected_model = model_name or model or CHAT_MODEL
    selected_temperature = float(TEMPERATURE if temperature is None else temperature)

    return _get_cached_llm(
        model_name=selected_model,
        temperature=selected_temperature,
    )


@lru_cache(maxsize=16)
def _get_cached_llm(
    model_name: str,
    temperature: float,
) -> ChatOpenAI:
    validate_llm_config()

    return ChatOpenAI(
        model=model_name,
        api_key=DASHSCOPE_API_KEY,
        base_url=QWEN_BASE_URL,
        temperature=temperature,
    )
