"""Lane router + LLM factory.

WHERE THIS SITS: between the agent and a concrete LLM client. The rest of the
codebase holds ONE object — a ``LaneRouter`` — and never touches a provider
directly. The router does two small jobs:

  1. It is a thin pass-through that implements the same ``LLMClient`` contract
     (base.py) and forwards every call to whatever client was wired underneath.
     So "which model backend" is decided once, here, and is invisible upstream.
  2. It owns the policy of mapping a *task name* to a *lane* (``choose_lane``):
     route/rewrite/summarize -> FAST, answer -> STANDARD, deep -> DEEP. (Today
     callers mostly pass the lane explicitly; this is the central place that
     could choose for them.)

``build_llm(settings)`` is the factory: read config, construct the matching
client (an OpenAI-compatible HTTP client pointed at SGLang or OpenAI), and wrap it
in a LaneRouter. This is the single seam where a deployment swaps its LLM backend.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from app.config import Settings
from app.core.llm.base import ChatMessage, Lane, LLMClient, LLMResponse, LLMToolSpec


class LaneRouter:
    """A wrapper that IS an LLMClient (same four methods) but delegates each to
    the wrapped ``client``. Holding the router instead of the raw client keeps
    the backend swappable and gives one place to add cross-cutting concerns
    (lane policy, retries, metrics, tracing) without touching the agent."""

    def __init__(self, client: LLMClient, tracer=None) -> None:
        self.client = client
        self.provider = getattr(client, "provider", "unknown")
        self._tracer = tracer

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
        with self._span("llm.complete", lane=lane.value):
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
        with self._span("llm.tools", lane=lane.value, n_tools=len(tools)):
            return await self.client.complete_with_tools(
                messages, tools, lane=lane, tool_choice=tool_choice, **kw
            )

    def _span(self, name: str, **attrs):
        if self._tracer:
            try:
                return self._tracer.span(name, **attrs)
            except Exception:  # noqa: BLE001
                pass
        from contextlib import nullcontext
        return nullcontext()

    async def aclose(self) -> None:
        await self.client.aclose()


def build_llm(settings: Settings, tracer=None) -> LaneRouter:
    """Construct the configured LLM client and wrap it in a LaneRouter.

    The ``llm_provider`` setting selects one of three real backends (sglang / vllm /
    openai). The tracer (Langfuse or NoOp) is passed through so every LLM call is
    automatically traced.
    """
    provider = settings.llm_provider
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
        return LaneRouter(client, tracer=tracer)

    if provider == "vllm":
        client = OpenAICompatClient(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            lane_models={lane: settings.vllm_model for lane in Lane},
            provider="vllm",
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
        return LaneRouter(client, tracer=tracer)

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
        return LaneRouter(client, tracer=tracer)

    from app.core.errors import ConfigError
    raise ConfigError(f"unknown llm_provider: {provider}")


__all__ = ["LaneRouter", "build_llm"]
