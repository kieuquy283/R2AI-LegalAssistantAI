from __future__ import annotations

import os
from importlib import import_module
from typing import Any

DEFAULT_CHAT_PIPELINE_MODE = "legacy"
SUPPORTED_CHAT_PIPELINE_MODES = {"adaptive", "legacy", "modular", "full"}


def get_chat_pipeline_mode(value: str | None = None) -> str:
    raw_value = (
        value
        or os.getenv("RAG_PIPELINE_MODE")
        or os.getenv("RAG_CHAT_PIPELINE")
        or DEFAULT_CHAT_PIPELINE_MODE
    )
    mode = str(raw_value).strip().lower()
    if mode == "full":
        return "modular"
    if mode not in SUPPORTED_CHAT_PIPELINE_MODES:
        return DEFAULT_CHAT_PIPELINE_MODE
    return mode


def _get_legacy_pipeline_cls():
    return getattr(import_module("rag.pipelines.chat_pipeline"), "ChatPipeline")


def _get_modular_pipeline_cls():
    return getattr(import_module("rag.pipelines.modular_chat_pipeline"), "ModularChatPipeline")


def _get_adaptive_pipeline_cls():
    return getattr(import_module("rag.pipelines.adaptive_modular_pipeline"), "AdaptiveModularPipeline")


def build_chat_pipeline(
    index_dir: str = "indexes/default",
    *,
    pipeline_mode: str | None = None,
    **kwargs: Any,
) -> Any:
    mode = get_chat_pipeline_mode(pipeline_mode)
    if mode == "adaptive":
        return _get_adaptive_pipeline_cls()(index_dir=index_dir, **kwargs)
    if mode == "modular":
        return _get_modular_pipeline_cls()(index_dir=index_dir, **kwargs)
    return _get_legacy_pipeline_cls()(index_dir=index_dir, **kwargs)
