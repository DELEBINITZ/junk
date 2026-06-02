"""Lane router + LLM factory.

``LaneRouter`` is the single handle the agent uses; it forwards to the configured
client and picks a lane per task (route/summarize -> fast, answer -> standard,
deep analysis -> deep). ``build_llm(settings)`` wires the provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from app.config import Settings
from app.core.llm.base import ChatMessage, Lane, LLMClient, LLMResponse, LLMToolSpec


class LaneRouter:
    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.provider = getattr(client, "provider", "unknown")

    @staticmethod
    def choose_lane(task: str) -> Lane:
        return {
            "route": Lane.FAST,
            "rewrite": Lane.FAST,
            "summarize": Lane.FAST,
            "answer": Lane.STANDARD,
            "deep": Lane.DEEP,
        }.get(task, Lane.STANDARD)

    async def complete(
        self, messages: Sequence[ChatMessage], *, lane: Lane = Lane.STANDARD, **kw
    ) -> LLMResponse:
        return await self.client.complete(messages, lane=lane, **kw)

    def stream(
        self, messages: Sequence[ChatMessage], *, lane: Lane = Lane.STANDARD, **kw
    ) -> AsyncIterator[str]:
        return self.client.stream(messages, lane=lane, **kw)

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[LLMToolSpec],
        *,
        lane: Lane = Lane.STANDARD,
        tool_choice: str = "auto",
        **kw,
    ) -> LLMResponse:
        return await self.client.complete_with_tools(
            messages, tools, lane=lane, tool_choice=tool_choice, **kw
        )

    async def aclose(self) -> None:
        await self.client.aclose()


def build_llm(settings: Settings) -> LaneRouter:
    provider = settings.llm_provider
    if provider == "deterministic":
        from app.core.llm.deterministic import DeterministicLLM

        return LaneRouter(DeterministicLLM(settings))

    from app.core.llm.openai_compat import OpenAICompatClient

    if provider == "sglang":
        client = OpenAICompatClient(
            base_url=settings.sglang_base_url,
            api_key=settings.sglang_api_key,
            lane_models={
                Lane.FAST: settings.model_fast,
                Lane.STANDARD: settings.model_standard,
                Lane.DEEP: settings.model_deep,
            },
            provider="sglang",
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
        return LaneRouter(client)

    if provider == "openai":
        client = OpenAICompatClient(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            lane_models={lane: settings.openai_model for lane in Lane},
            provider="openai",
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
        return LaneRouter(client)

    from app.core.errors import ConfigError

    raise ConfigError(f"unknown llm_provider: {provider}")


__all__ = ["LaneRouter", "build_llm"]
