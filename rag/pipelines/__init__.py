from __future__ import annotations

from importlib import import_module

__all__ = [
    "ChatPipeline",
    "build_chat_pipeline",
    "DEFAULT_CHAT_PIPELINE_MODE",
    "get_chat_pipeline_mode",
    "LegacyCompatibleModularChatPipeline",
    "ModularChatPipeline",
]


def __getattr__(name: str):
    if name == "ChatPipeline":
        return getattr(import_module("rag.pipelines.chat_pipeline"), name)
    if name in {"build_chat_pipeline", "DEFAULT_CHAT_PIPELINE_MODE", "get_chat_pipeline_mode"}:
        return getattr(import_module("rag.pipelines.factory"), name)
    if name in {"ModularChatPipeline", "LegacyCompatibleModularChatPipeline"}:
        return getattr(import_module("rag.pipelines.modular_chat_pipeline"), name)
    raise AttributeError(f"module 'rag.pipelines' has no attribute {name!r}")
