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
    (lane policy, retries, metrics) without touching the agent."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        # Re-expose the underlying provider name so callers can still branch on
        # it (e.g. "is this the deterministic stub?") through the router.
        self.provider = getattr(client, "provider", "unknown")

    @staticmethod
    def choose_lane(task: str) -> Lane:
        """Map a logical task to the cheapest lane that can do it well. Cheap,
        high-volume tasks (routing, rewriting, summarizing) go to FAST; the
        user-facing answer goes to STANDARD; explicitly hard work goes to DEEP.
        Unknown tasks default to STANDARD — a safe middle tier."""
        return {
            "route": Lane.FAST,
            "rewrite": Lane.FAST,
            "summarize": Lane.FAST,
            "answer": Lane.STANDARD,
            "deep": Lane.DEEP,
        }.get(task, Lane.STANDARD)

    # The four methods below are pure delegation — they exist so the router
    # satisfies the LLMClient contract while forwarding to the wrapped client.
    async def complete(
        self, messages: Sequence[ChatMessage], *, lane: Lane = Lane.STANDARD, **kw
    ) -> LLMResponse:
        return await self.client.complete(messages, lane=lane, **kw)

    # Forwarded WITHOUT await on purpose: stream() returns an async iterator, so
    # we return the client's iterator directly for the caller to ``async for``.
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
        # Propagate shutdown to the real client so its HTTP connections close.
        await self.client.aclose()


def build_llm(settings: Settings) -> LaneRouter:
    """Construct the configured LLM client and wrap it in a LaneRouter.

    The ``llm_provider`` setting selects one of three real backends (sglang / vllm /
    openai). The HTTP client import is deferred so it's only loaded at build time.
    """
    provider = settings.llm_provider
    # All providers speak the SAME OpenAI-compatible HTTP API, so they share one
    # client class and differ only in URL/keys and the lane->model map.
    from app.core.llm.openai_compat import OpenAICompatClient

    if provider == "sglang":
        # SGLang = our self-hosted serving stack. Each lane points at a DIFFERENT
        # served model (the 7B/72B/DeepSeek mapping from base.py), so a single
        # endpoint exposes all three tiers.
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

    if provider == "vllm":
        # vLLM (self-hosted, OpenAI-compatible) typically serves ONE model per
        # server (e.g. a 32B for staging), so all three lanes map to that single
        # ``vllm_model`` — the tiers collapse, same as the openai case.
        client = OpenAICompatClient(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            lane_models={lane: settings.vllm_model for lane in Lane},
            provider="vllm",
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
        return LaneRouter(client)

    if provider == "openai":
        # Optional dev convenience: talk to OpenAI (or any OpenAI-compatible
        # endpoint). Here ALL three lanes map to the same single model — the
        # tiers collapse because we are not self-hosting three separate models.
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

    # A misconfigured provider name is a fail-fast config error at boot, not a
    # silent fallback — better to refuse to start than serve with the wrong LLM.
    from app.core.errors import ConfigError

    raise ConfigError(f"unknown llm_provider: {provider}")


__all__ = ["LaneRouter", "build_llm"]
