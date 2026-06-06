from __future__ import annotations

import os
from typing import Optional


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.temperature = float(temperature)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, *, system_prompt: str, user_prompt: str, temperature: float | None = None) -> Optional[str]:
        if not self.is_available():
            return None
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.responses.create(
                model=self.model,
                temperature=float(self.temperature if temperature is None else temperature),
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            return None

        text = getattr(response, "output_text", None)
        return str(text).strip() if text else None
