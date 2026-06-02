"""LLM client factory.

The default client is deterministic so tests and demos do not depend on a model
server. When configured, Ollama or vLLM participates in grounded RAG answer
generation or final wording, but facts and security decisions still come from
backend tools and guardrails.
"""

from __future__ import annotations

from app.config import settings


class LLMClient:
    provider_name = "unknown"
    model_name = "unknown"

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """Invoke a chat-style local LLM with explicit system/user prompts."""

        raise NotImplementedError

    def complete(self, prompt: str) -> str:
        """Backward-compatible single-prompt completion helper."""

        return self.invoke(system_prompt="", user_prompt=prompt)


class DeterministicLLMClient(LLMClient):
    """No-op client used for repeatable local behavior."""

    provider_name = "deterministic"
    model_name = "none"

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        return user_prompt


def get_llm_client() -> LLMClient:
    """Return the configured local/self-hosted LLM client."""

    provider = settings.llm_provider.lower()
    if provider == "ollama":
        from app.llm.ollama_client import OllamaClient

        return OllamaClient()
    if provider == "vllm":
        from app.llm.vllm_client import VLLMClient

        return VLLMClient()
    return DeterministicLLMClient()


def estimate_tokens(text: str) -> int:
    """Return a lightweight token estimate for reporting and demos.

    This intentionally does not depend on a tokenizer package. It is close
    enough for usage dashboards and makes clear that deterministic/offline paths
    are reporting estimated, not provider-billed, token counts.
    """

    if not text:
        return 0
    return max(1, round(len(text.split()) * 1.33))
