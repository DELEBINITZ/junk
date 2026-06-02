"""LLM client contract + shared message/response types.

Three lanes (fast / standard / deep) map to three self-hosted models in prod
(Qwen 7B / Qwen 72B / DeepSeek). The contract is provider-agnostic; concrete
clients live alongside (``deterministic``, ``sglang_client``, ``openai_client``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class Lane(str, Enum):
    FAST = "fast"          # routing, rewriting, summarizing  (Qwen 7B)
    STANDARD = "standard"  # default answering               (Qwen 72B)
    DEEP = "deep"          # hard multi-hop analysis          (DeepSeek)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


class LLMToolSpec(BaseModel):
    """Function-calling advertisement passed to the model."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


class LLMToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    lane: Lane = Lane.STANDARD
    usage: LLMUsage = Field(default_factory=LLMUsage)


class LLMClient(Protocol):
    provider: str

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        lane: Lane = Lane.STANDARD,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        lane: Lane = Lane.STANDARD,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[LLMToolSpec],
        *,
        lane: Lane = Lane.STANDARD,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    async def aclose(self) -> None: ...


CONTEXT_MARKER = "RETRIEVED CONTEXT"
NO_CONTEXT_REFUSAL = (
    "I don't have enough grounded information in the retrieved sources to answer "
    "that confidently. Try narrowing the question or check that the relevant "
    "report has been ingested."
)


__all__ = [
    "Lane",
    "ChatMessage",
    "LLMToolSpec",
    "LLMToolCall",
    "LLMUsage",
    "LLMResponse",
    "LLMClient",
    "CONTEXT_MARKER",
    "NO_CONTEXT_REFUSAL",
]
