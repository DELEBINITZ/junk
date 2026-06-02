"""Agent turn state + the context bundle nodes operate on.

State is a plain ``dict`` (typed by :class:`ChatState`) so the SAME node
functions run under both the built-in engine and real LangGraph — nodes take a
state dict and return a partial-update dict, which each engine merges.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from app.core.contracts import Chunk, CoreDeps, ToolContext
from app.core.security.context import SecurityContext

# Node names (single source of truth; shared by both engines).
N_INPUT_GUARD = "input_guardrail"
N_ROUTE = "route"
N_GATHER = "gather_context"
N_ANSWER = "answer"
N_OUTPUT_GUARD = "output_guardrail"


class ChatState(TypedDict, total=False):
    # identity / request
    org_id: str
    user_id: str
    roles: tuple[str, ...]
    session_id: str
    trace_id: str
    request_id: str
    # input
    question: str
    history: list[dict[str, Any]]
    summary: str
    # guardrail
    safe_question: str
    blocked: bool
    block_reason: str
    # routing
    route_modules: list[str]
    route_debug: dict[str, Any]
    # retrieval / tools
    context_chunks: list[Chunk]
    context_block: str
    tool_events: list[dict[str, Any]]
    # answer
    answer: str
    citations: list[dict[str, Any]]
    output_flags: dict[str, Any]
    lane: str
    error: str


def make_initial_state(
    *,
    sc: SecurityContext,
    question: str,
    session_id: str,
    trace_id: str,
    request_id: str,
    history: list[dict[str, Any]] | None = None,
    summary: str = "",
) -> ChatState:
    return ChatState(
        org_id=sc.org_id, user_id=sc.user_id, roles=sc.roles, session_id=session_id,
        trace_id=trace_id, request_id=request_id, question=question,
        history=history or [], summary=summary, blocked=False,
        route_modules=[], context_chunks=[], context_block="", tool_events=[],
        answer="", citations=[], output_flags={},
    )


# Streaming event passed to ctx.emit (mirrors SSE event types).
@dataclass
class AgentEvent:
    type: str                       # status | route | tool | token | citation | error | done
    data: dict[str, Any] = field(default_factory=dict)


EmitFn = Callable[[AgentEvent], Awaitable[None]]


@dataclass
class AgentContext:
    """Everything nodes need — services, identity, and an optional event sink."""

    deps: CoreDeps
    sc: SecurityContext
    tool_ctx: ToolContext
    mcp: Any                        # MCPClient
    registry: Any                   # CapabilityRegistry
    input_guard: Any
    output_guard: Any
    settings: Any
    supervisor: Any = None          # app.core.agent.supervisor.Supervisor
    emit: EmitFn | None = None
    stream_tokens: bool = False

    async def fire(self, type: str, **data: Any) -> None:
        if self.emit is not None:
            await self.emit(AgentEvent(type=type, data=data))


__all__ = [
    "ChatState",
    "AgentContext",
    "AgentEvent",
    "EmitFn",
    "make_initial_state",
    "N_INPUT_GUARD",
    "N_ROUTE",
    "N_GATHER",
    "N_ANSWER",
    "N_OUTPUT_GUARD",
]
