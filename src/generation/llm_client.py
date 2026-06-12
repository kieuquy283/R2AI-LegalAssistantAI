from __future__ import annotations

import os
import time
from typing import Optional


class LLMClient:
    """LLM client supporting both OpenAI and DashScope (Qwen)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        # Prefer DashScope/Qwen if available
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL")
        self.model = model or os.getenv("CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.temperature = float(temperature)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, *, system_prompt: str, user_prompt: str, temperature: float | None = None) -> Optional[str]:
        if not self.is_available():
            return None
        try:
            from openai import OpenAI

            if self.base_url:
                # DashScope mode (OpenAI-compatible API)
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                # Pure OpenAI mode
                client = OpenAI(api_key=self.api_key)

            temp = float(self.temperature if temperature is None else temperature)
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=self.model,
                temperature=temp,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            latency = time.perf_counter() - t0
            print(f"[LLM] {self.model} response in {latency:.2f}s")
        except Exception as exc:
            print(f"[LLM] API call failed: {exc}")
            return None

        text = getattr(response.choices[0].message, "content", None) if response.choices else None
        return str(text).strip() if text else None
