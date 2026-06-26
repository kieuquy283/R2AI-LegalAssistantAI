from __future__ import annotations

import os
import time
from typing import Optional


_MAX_RETRIES = 2
_BASE_DELAY = 2.0


class LLMClient:
    """LLM client with multi-key fallback for rate limit handling.

    Features:
    - Automatic key rotation on 429 rate limits
    - Exponential backoff between retries
    - Supports GROQ_API_KEY_2, OPENROUTER_API_KEY_2, HF_TOKEN_2
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        self.temperature = float(temperature)

        # Collect all available API keys for this provider
        self._api_keys: list[str] = []
        self._key_idx = 0

        if self.provider == "hf":
            self.hf_model = model or os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip()
            for env in ["HF_TOKEN", "HF_TOKEN_2"]:
                k = os.getenv(env, "").strip()
                if k:
                    self._api_keys.append(k)
            self.api_key = self._api_keys[0] if self._api_keys else ""
        else:
            self.base_url = base_url or os.getenv("QWEN_BASE_URL")
            self.model = model or os.getenv("CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
            # Collect keys based on base_url
            if self.base_url and "groq" in self.base_url:
                for i in range(1, 20):
                    env = "GROQ_API_KEY" if i == 1 else f"GROQ_API_KEY_{i}"
                    k = os.getenv(env, "").strip()
                    if k:
                        self._api_keys.append(k)
            elif self.base_url and "openrouter" in self.base_url:
                for i in range(1, 20):
                    env = "OPENROUTER_API_KEY" if i == 1 else f"OPENROUTER_API_KEY_{i}"
                    k = os.getenv(env, "").strip()
                    if k:
                        self._api_keys.append(k)
            else:
                # Generic: try all key envs
                for env in ["OPENAI_API_KEY", "DASHSCOPE_API_KEY"]:
                    k = os.getenv(env, "").strip()
                    if k:
                        self._api_keys.append(k)

            if not self._api_keys and api_key:
                self._api_keys.append(api_key)
            self.api_key = self._api_keys[0] if self._api_keys else ""

    def _rotate_key(self) -> str:
        """Rotate to next available API key. Returns the new key."""
        if len(self._api_keys) <= 1:
            return self.api_key
        self._key_idx = (self._key_idx + 1) % len(self._api_keys)
        self.api_key = self._api_keys[self._key_idx]
        print(f"[LLM] Rotated to API key #{self._key_idx + 1} (...{self.api_key[-8:]})")
        return self.api_key

    def is_available(self) -> bool:
        return bool(self._api_keys)

    def _is_rate_limited(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(code in msg for code in ("429", "rate limit", "too many requests", "tokens per day", "tpd"))

    def _generate_hf(self, *, system_prompt: str, user_prompt: str, temperature: float) -> Optional[str]:
        if not self.api_key:
            return None

        def _call():
            from huggingface_hub import InferenceClient
            client = InferenceClient(api_key=self.api_key)
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

        last_exc = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return _call()
            except Exception as exc:
                last_exc = exc
                if self._is_rate_limited(exc) and len(self._api_keys) > 1:
                    self._rotate_key()
                    time.sleep(1.0)
                    continue
                if not self._is_rate_limited(exc):
                    raise
                delay = _BASE_DELAY * (2 ** attempt)
                print(f"[LLM] Rate limited (attempt {attempt + 1}), retrying in {delay:.1f}s...")
                time.sleep(delay)
        print(f"[LLM] All retries exhausted: {last_exc}")
        return None

    def _generate_openai(self, *, system_prompt: str, user_prompt: str, temperature: float) -> Optional[str]:
        if not self.api_key:
            return None

        def _call():
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30.0)
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            latency = time.perf_counter() - t0
            print(f"[LLM] {self.model} response in {latency:.2f}s")
            text = getattr(response.choices[0].message, "content", None) if response.choices else None
            return str(text).strip() if text else None

        last_exc = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return _call()
            except Exception as exc:
                last_exc = exc
                if self._is_rate_limited(exc) and len(self._api_keys) > 1:
                    self._rotate_key()
                    time.sleep(1.0)
                    continue
                if not self._is_rate_limited(exc):
                    raise
                delay = _BASE_DELAY * (2 ** attempt)
                print(f"[LLM] Rate limited (attempt {attempt + 1}), retrying in {delay:.1f}s...")
                time.sleep(delay)
        print(f"[LLM] All retries exhausted: {last_exc}")
        return None

    def generate(self, *, system_prompt: str, user_prompt: str, temperature: float | None = None) -> Optional[str]:
        if not self.is_available():
            return None
        temp = float(self.temperature if temperature is None else temperature)
        if self.provider == "hf":
            return self._generate_hf(system_prompt=system_prompt, user_prompt=user_prompt, temperature=temp)
        return self._generate_openai(system_prompt=system_prompt, user_prompt=user_prompt, temperature=temp)
