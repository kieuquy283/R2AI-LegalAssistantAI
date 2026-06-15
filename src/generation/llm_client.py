from __future__ import annotations

import os
import time
from typing import Optional


class LLMClient:
    """LLM client supporting HuggingFace Inference API, OpenAI, and DashScope (Qwen)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        # Determine LLM provider
        self.provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        
        # HuggingFace Inference API settings
        self.hf_token = api_key or os.getenv("HF_TOKEN", "").strip()
        self.hf_model = model or os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip()
        
        # OpenAI/DashScope settings
        self.api_key = self.api_key if hasattr(self, 'api_key') else (api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"))
        self.base_url = base_url or os.getenv("QWEN_BASE_URL")
        self.model = model or os.getenv("CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        
        self.temperature = float(temperature)

    def is_available(self) -> bool:
        if self.provider == "hf":
            return bool(self.hf_token)
        return bool(self.api_key)

    def _generate_hf(self, *, system_prompt: str, user_prompt: str, temperature: float) -> Optional[str]:
        """Generate response using HuggingFace Inference API via huggingface_hub."""
        if not self.hf_token:
            return None

        try:
            from huggingface_hub import InferenceClient
            
            client = InferenceClient(api_key=self.hf_token)
            
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=self.hf_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=max(temperature, 0.1),
                max_tokens=2048,
            )
            latency = time.perf_counter() - t0
            print(f"[LLM] HF {self.hf_model} response in {latency:.2f}s")
            
            text = getattr(response.choices[0].message, "content", None) if response.choices else None
            return str(text).strip() if text else None
            
        except Exception as e:
            print(f"[LLM] HF API call failed: {e}")
            return None

    def generate(self, *, system_prompt: str, user_prompt: str, temperature: float | None = None) -> Optional[str]:
        if not self.is_available():
            return None
        
        temp = float(self.temperature if temperature is None else temperature)
        
        # Use HuggingFace Inference API if provider is hf
        if self.provider == "hf":
            return self._generate_hf(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temp,
            )
        
        # Use OpenAI/DashScope API
        try:
            from openai import OpenAI

            if self.base_url:
                # DashScope mode (OpenAI-compatible API)
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                # Pure OpenAI mode
                client = OpenAI(api_key=self.api_key)

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
