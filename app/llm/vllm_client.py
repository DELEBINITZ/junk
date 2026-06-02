"""vLLM OpenAI-compatible local LLM client."""

from __future__ import annotations

import httpx

from app.config import settings
from app.llm.client import LLMClient


class VLLMClient(LLMClient):
    """Call a local vLLM server through the OpenAI-compatible chat API."""

    provider_name = "vllm"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self.model = model or settings.vllm_model
        self.model_name = self.model

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """Call vLLM's OpenAI-compatible chat API with explicit messages."""

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
            },
            timeout=60,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])
