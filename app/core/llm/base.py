"""LLM client contract + shared message/response types.

WHAT THIS FILE IS: the single, provider-agnostic interface every part of the
agent talks to when it wants an LLM. Nodes and specialists never import a
concrete client (OpenAI, SGLang, the stub); they depend on the ``LLMClient``
Protocol declared here, so the backend can be swapped purely by config.

MENTAL MODEL — the three "lanes". Instead of one model for everything, the
system splits work across three quality/cost tiers and asks each task to run on
the cheapest lane that can do the job well:

    FAST     -> cheap + quick  : routing, query rewriting, summarizing, tool-planning
    STANDARD -> the default    : writing the user-facing answer
    DEEP     -> slow + strong  : hard multi-hop analysis

In production each lane maps to a different self-hosted model (Qwen 7B /
Qwen 72B / DeepSeek). "Self-hosted" here means we run the weights ourselves
(via SGLang) rather than calling a vendor API — important for a security
product where customer data must not leave our infrastructure.

WHY A PROTOCOL (not a base class): a Protocol is structural ("duck") typing —
any object that has these methods *is* an LLMClient, with no inheritance
required. That lets the deterministic stub, the OpenAI-compatible HTTP client,
and the lane router all satisfy the same contract independently. The concrete
clients live alongside this file (``deterministic``, ``openai_compat``) and are
selected in ``lanes.build_llm``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class Lane(str, Enum):
    """The three model tiers (see the module mental model). Inheriting from
    ``str`` makes a Lane *be* its string value ("fast"), so it serializes into
    JSON/logs and compares as a plain string with no conversion."""

    FAST = "fast"          # routing, rewriting, summarizing  (Qwen 7B)
    STANDARD = "standard"  # default answering               (Qwen 72B)
    DEEP = "deep"          # hard multi-hop analysis          (DeepSeek)


class ChatMessage(BaseModel):
    """One message in a chat conversation, modelled on the OpenAI chat format
    (a list of ``{role, content}`` messages is the universal shape every modern
    chat LLM expects). ``role`` is who is "speaking":
      * system    — standing instructions / context (the persona, the retrieved
                    sources). The model treats this as setup, not as a question.
      * user      — what the human asked.
      * assistant — what the model previously replied (for multi-turn history).
      * tool      — the result of a tool call fed back to the model.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None  # links a tool result back to the call that produced it

    def to_openai(self) -> dict[str, Any]:
        """Render to the plain dict the OpenAI-compatible HTTP API expects.
        Optional fields are only included when set, keeping the wire payload
        minimal and matching what servers like SGLang accept."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


class LLMToolSpec(BaseModel):
    """Function-calling advertisement passed to the model.

    "Function calling" (a.k.a. tool use) is how an LLM asks to run code: you
    describe the available tools — name, what it does, and a JSON Schema of its
    arguments — and the model, instead of writing prose, replies with a
    structured request to call one (see ``LLMToolCall``). The ``description`` and
    ``parameters`` are what the model reads to decide *whether* and *how* to call
    a tool, so they double as the tool's prompt."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON Schema of the args

    def to_openai(self) -> dict[str, Any]:
        """Wrap into the OpenAI ``tools=[...]`` shape the chat API expects."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


class LLMToolCall(BaseModel):
    """The model's REQUEST to call a tool — the other side of function calling.
    The agent does not run this directly; it routes the call through the MCP
    boundary (mcp/inprocess.py) where RBAC and the action gate are enforced."""

    id: str = ""                                              # echoed back when returning the tool result
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMUsage(BaseModel):
    """Token accounting returned by the server — drives cost/latency tracking."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """The normalized result of a completion, identical across every provider.
    Either ``text`` is filled (a normal answer) or ``tool_calls`` is non-empty
    (the model wants to call tools); ``finish_reason`` says why generation
    stopped. Normalizing here means callers never parse raw provider JSON."""

    text: str = ""
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    lane: Lane = Lane.STANDARD  # which tier actually served this, for observability
    usage: LLMUsage = Field(default_factory=LLMUsage)


class LLMClient(Protocol):
    """THE CONTRACT. Every concrete client implements exactly these four methods,
    so the agent can be wired to any backend without code changes. Three ways to
    generate, plus a cleanup hook:

      * ``complete``           — one-shot: send messages, get the whole answer back.
      * ``stream``             — same, but yields the answer token-by-token as it is
                                 produced (so the UI can show it appearing live).
      * ``complete_with_tools``— advertise tools and let the model pick which to
                                 call (function calling). Used by the specialist's
                                 LLM planner.
      * ``aclose``             — release any held resources (e.g. an HTTP client).

    ``provider`` is a short string ("deterministic" / "sglang" / "openai") that
    callers occasionally branch on — e.g. the specialist skips the LLM tool
    planner when the provider is the deterministic stub.
    """

    provider: str

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        lane: Lane = Lane.STANDARD,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    # Note this is a plain ``def`` (not ``async def``): it RETURNS an async
    # iterator that callers ``async for`` over. Each yielded str is one streamed
    # token/delta. Declaring it this way lets the lane router forward the
    # iterator straight through without itself becoming a coroutine.
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


# The fixed label that prefixes the retrieved-context system message in every
# answer prompt (see nodes.build_answer_messages). Keeping it a single shared
# constant means the prompt format is consistent and the deterministic stub can
# reliably FIND the context block to parse it (see deterministic.py).
CONTEXT_MARKER = "RETRIEVED CONTEXT"
# The canonical "I won't guess" message. Central to the system being a RAG
# assistant and not a chatbot: when nothing relevant was retrieved, the honest
# answer is to refuse rather than hallucinate. Reused so the refusal wording is
# identical everywhere it can occur.
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
