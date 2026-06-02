"""Ollama-backed local LLM client."""

from __future__ import annotations

import httpx

from app.config import settings
from app.llm.client import LLMClient


class OllamaClient(LLMClient):
    """Call a locally served Ollama model for grounded RAG generation."""

    provider_name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.model_name = self.model

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama's chat API with explicit system and user messages."""

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return str(response.json().get("message", {}).get("content", ""))
