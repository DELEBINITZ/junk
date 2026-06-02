"""SGLang client (OpenAI-compatible /chat/completions), with token streaming.

Self-hosted answerer transport (plan §14). Uses httpx (already a dependency).
Reachability is only exercised when LLM_PROVIDER=sglang; construction does no I/O.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

import httpx

from app.llm.client import LLMClient


logger = logging.getLogger(__name__)


class SGLangClient(LLMClient):
    provider_name = "sglang"

    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.timeout = timeout

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": self._messages(system_prompt, user_prompt),
            "temperature": 0,
        }
        response = httpx.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield content deltas as they arrive (real per-token streaming)."""

        payload = {
            "model": self.model_name,
            "messages": self._messages(system_prompt, user_prompt),
            "temperature": 0,
            "stream": True,
        }
        with httpx.stream("POST", f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                if delta:
                    yield delta
