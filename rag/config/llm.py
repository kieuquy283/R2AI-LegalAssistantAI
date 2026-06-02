from __future__ import annotations

import os

from rag.config.paths import ENV_PATH  

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
).strip()

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus").strip()
REWRITE_MODEL = os.getenv("REWRITE_MODEL", CHAT_MODEL).strip()
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))


def validate_llm_config() -> None:
    if not DASHSCOPE_API_KEY:
        raise ValueError(
            "Không tìm thấy DASHSCOPE_API_KEY trong .env. "
            "Hãy tạo/cập nhật file .env ở root project."
        )