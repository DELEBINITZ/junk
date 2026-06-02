"""Agent turn state + the context bundle nodes operate on.

Two distinct things live here — keep them apart in your head:

1. ``ChatState`` — the DATA that flows THROUGH the graph (graph.py). It is a
   plain ``dict`` describing one chat turn: the question, retrieved context, the
   answer, etc. Each node reads it and returns a few changed keys, which the
   engine merges in. It is a ``dict`` (not a class) on purpose: the SAME node
   functions must run under both our built-in engine AND real LangGraph, and
   both speak "dict in, partial-dict out".

2. ``AgentContext`` — the SERVICES + identity a node needs to DO its work (the
   LLM, the retrieval pipeline, the registry, the logged-in user, an event
   sink for streaming). This does NOT flow as graph state; it is bound to each
   node via a closure when the graph is built (see nodes.build_report_graph).

Rule of thumb: if it changes during the turn -> ChatState. If it is a service
or fixed identity for the whole turn -> AgentContext.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from app.core.contracts import Chunk, CoreDeps, ToolContext
from app.core.security.context import SecurityContext

# Node names — defined ONCE here and imported everywhere, so the built-in engine
# (nodes.py) and the LangGraph engine (engines.py) wire the SAME graph. If these
# were stray string literals, the two engines could drift out of sync.
N_INPUT_GUARD = "input_guardrail"
N_ROUTE = "route"
N_GATHER = "gather_context"
N_ANSWER = "answer"
N_OUTPUT_GUARD = "output_guardrail"


class ChatState(TypedDict, total=False):
    """The per-turn state dict. ``total=False`` means every key is OPTIONAL —
    nodes fill keys in progressively as the turn advances (the guardrail adds
    ``safe_question``, routing adds ``route_modules``, retrieval adds
    ``context_chunks``, and so on). Think of it as a form that gets filled out
    section by section as it passes down the graph.

    The comments below group keys by the node that produces them.
    """

    # --- identity / request (set once at the start, never changed) ---
    org_id: str          # the tenant — copied from the verified token, the isolation key
    user_id: str
    roles: tuple[str, ...]
    session_id: str
    trace_id: str        # correlates all logs/events for this one turn
    request_id: str
    # --- input (set at the start) ---
    question: str                    # the raw user message
    history: list[dict[str, Any]]    # prior turns in this session (for context)
    summary: str                     # rolling summary of older turns (bounds context size)
    # --- produced by input_guardrail_node ---
    safe_question: str   # the question after redaction/screening (what we actually use)
    blocked: bool        # True => guardrail refused; engine jumps to END
    block_reason: str
    # --- produced by route_node (the supervisor) ---
    route_modules: list[str]          # which capability module(s) handle this turn
    route_debug: dict[str, Any]       # scores/mode/fallback — for /route/preview + logs
    # --- produced by dispatch_node (specialists + their tools/retrieval) ---
    context_chunks: list[Chunk]       # the retrieved evidence, ranked + capped
    context_block: str                # those chunks rendered as "[1] ... [2] ..." for the prompt
    tool_events: list[dict[str, Any]] # trace of which tools ran (observability)
    # --- produced by answer_node ---
    answer: str
    citations: list[dict[str, Any]]   # [n] markers mapped back to their source chunks
    output_flags: dict[str, Any]      # groundedness / guardrail flags
    lane: str                         # which LLM lane answered (fast/standard/deep)
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
    """Build the starting state for a turn. Only identity + input keys are set;
    everything the nodes produce starts empty/false. Notice ``org_id`` etc. come
    from ``sc`` (the SecurityContext built from the verified token) — NEVER from
    anything the user typed. That is the root of tenant isolation."""
    return ChatState(
        org_id=sc.org_id, user_id=sc.user_id, roles=sc.roles, session_id=session_id,
        trace_id=trace_id, request_id=request_id, question=question,
        history=history or [], summary=summary, blocked=False,
        route_modules=[], context_chunks=[], context_block="", tool_events=[],
        answer="", citations=[], output_flags={},
    )


# A streaming EVENT. As the turn runs, nodes call ``ctx.fire(...)`` which wraps
# data in one of these and pushes it to the client over SSE (Server-Sent Events).
# ``type`` is the event kind the frontend switches on; ``data`` is its payload.
@dataclass
class AgentEvent:
    type: str                       # status | route | tool | token | citation | error | done
    data: dict[str, Any] = field(default_factory=dict)


# The shape of the "emit" callback an AgentContext may carry. ``None`` means the
# turn isn't streaming (e.g. the plain POST /chat path) so events are dropped.
EmitFn = Callable[[AgentEvent], Awaitable[None]]


@dataclass
class AgentContext:
    """Everything nodes need that ISN'T turn data: the core services bundle
    (``deps``), the caller's identity (``sc``), the trusted ``tool_ctx`` passed
    to tools, the MCP client + registry, the guardrails, the supervisor, and an
    optional event sink for streaming. One of these is built per request in the
    orchestrator and closed over by every node."""

    deps: CoreDeps                  # llm, rag, registry, conversations, kg, action_gate, ... (contracts.CoreDeps)
    sc: SecurityContext             # verified identity (org_id, user_id, roles)
    tool_ctx: ToolContext           # the trusted context handed to every tool call
    mcp: Any                        # MCPClient — the RBAC+gate-enforcing tool boundary
    registry: Any                   # CapabilityRegistry — the discovered modules
    input_guard: Any                # input guardrail spine
    output_guard: Any               # output guardrail spine
    settings: Any
    supervisor: Any = None          # app.core.agent.supervisor.Supervisor (the router)
    emit: EmitFn | None = None      # event sink; None when not streaming
    stream_tokens: bool = False     # True => answer_node streams the LLM token-by-token

    async def fire(self, type: str, **data: Any) -> None:
        """Emit a streaming event IF this turn is streaming (emit is set).
        Nodes call this freely; when not streaming it's a no-op, so the same node
        code serves both the streaming and non-streaming paths."""
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
