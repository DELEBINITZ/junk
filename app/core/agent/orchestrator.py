"""The ORCHESTRATOR — the chat service's entry point and turn owner.

If graph.py is "how an agent walks a graph" and nodes.py is "what each step
does", THIS file is "everything around one turn that isn't reasoning":

    load history + rolling summary  ->  persist the user message
        ->  build a per-request AgentContext  ->  run the graph (some engine)
        ->  persist the assistant message  ->  update the rolling summary
        ->  record a memory observation  ->  return / stream the result

So the split of responsibilities is: orchestrator = sessions, persistence,
streaming plumbing; graph nodes = the actual thinking. Keeping them apart is why
the nodes stay pure ``(state, ctx) -> partial state`` functions.

Two public methods:
  * run_turn   — non-streaming; returns a TurnResult (used by POST /chat).
  * stream_turn— streaming; yields AgentEvents over time (used by GET /chat/stream
    via SSE). It actually just calls run_turn with an ``emit`` callback wired to
    a queue, then drains that queue — so there is only ONE turn implementation.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.agent.engines import build_engine
from app.core.agent.state import AgentContext, AgentEvent, EmitFn, make_initial_state
from app.core.agent.supervisor import RouteResult
from app.core.contracts import ToolContext
from app.core.errors import NotFound
from app.core.security.context import SecurityContext

# Private marker pushed onto the stream queue to signal "the turn is finished".
# A unique object() can never collide with a real event.
_SENTINEL = object()


@dataclass
class TurnResult:
    """The structured result of one non-streaming turn (what POST /chat returns,
    after the API maps it to a response schema)."""
    answer: str
    citations: list[dict[str, Any]]
    route_modules: list[str]
    output_flags: dict[str, Any]
    trace_id: str
    session_id: str
    message_id: str
    route_debug: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    def __init__(
        self,
        *,
        settings,
        registry,
        deps,
        mcp,
        input_guard,
        output_guard,
        supervisor,
        conversations,   # the session/message store (memory or Postgres+RLS)
        summarizer,      # rolls long histories into a short summary
        kg,              # knowledge-graph memory seam (NoOp by default)
        checkpointer=None,
    ) -> None:
        # All services are INJECTED (built once in bootstrap.py and passed in).
        # The orchestrator creates nothing global — that's what makes it testable
        # and the whole app config-driven.
        self.settings = settings
        self.registry = registry
        self.deps = deps
        self.mcp = mcp
        self.input_guard = input_guard
        self.output_guard = output_guard
        self.supervisor = supervisor
        self.conversations = conversations
        self.summarizer = summarizer
        self.kg = kg
        self.checkpointer = checkpointer

    # -- helpers -------------------------------------------------------------
    def _tool_ctx(self, sc: SecurityContext, trace_id: str, request_id: str) -> ToolContext:
        """Build the TRUSTED context every tool receives. Critically, ``org_id``
        comes from ``sc`` (the verified token), so a tool's tenant is fixed by
        identity, never by anything in the request body or the model's args."""
        return ToolContext(
            org_id=sc.org_id, user_id=sc.user_id, roles=sc.roles,
            trace_id=trace_id, request_id=request_id, deps=self.deps,
        )

    def _agent_ctx(
        self, sc: SecurityContext, trace_id: str, request_id: str,
        emit: EmitFn | None, stream: bool,
    ) -> AgentContext:
        """Bundle everything the graph nodes need for THIS request into one
        AgentContext (see state.py). ``emit``/``stream`` decide whether nodes
        fire streaming events and whether the answer streams token-by-token."""
        return AgentContext(
            deps=self.deps, sc=sc, tool_ctx=self._tool_ctx(sc, trace_id, request_id),
            mcp=self.mcp, registry=self.registry, input_guard=self.input_guard,
            output_guard=self.output_guard, settings=self.settings, supervisor=self.supervisor,
            emit=emit, stream_tokens=stream,
        )

    async def _ensure_session(self, sc: SecurityContext, session_id: str | None):
        """Resume an existing chat session or start a new one. The lookup is
        org-scoped (``sc.org_id``), so you can only ever load YOUR org's session —
        a stranger's session id returns NotFound, not someone else's chat."""
        if session_id:
            s = await self.conversations.get_session(sc.org_id, session_id)
            if not s:
                raise NotFound(f"session '{session_id}' not found", details={"session_id": session_id})
            return s
        return await self.conversations.create_session(sc.org_id, sc.user_id)

    async def preview_route(self, sc: SecurityContext, question: str) -> RouteResult:
        """Expose the supervisor's decision WITHOUT running a turn — backs the
        /route/preview endpoint, handy for debugging routing."""
        return await self.supervisor.route(question, sc)

    # -- run -----------------------------------------------------------------
    async def run_turn(
        self,
        sc: SecurityContext,
        *,
        question: str,
        session_id: str | None = None,
        emit: EmitFn | None = None,
        stream: bool = False,
    ) -> TurnResult:
        """Execute ONE full chat turn end-to-end. This is the spine; read it as a
        numbered sequence."""
        # 1. Resolve the session and load prior context (history + rolling summary).
        session = await self._ensure_session(sc, session_id)
        prior = await self.conversations.get_messages(
            sc.org_id, session.id, limit=self.settings.history_window_messages
        )
        history = [{"role": m.role, "content": m.content} for m in prior]
        # 2. Persist the user's message immediately (before reasoning) so the
        #    conversation is durable even if generation fails midway.
        await self.conversations.append_message(sc.org_id, session.id, "user", question)

        # 3. Fresh ids for this turn: trace_id ties all its logs/events together.
        trace_id = uuid.uuid4().hex
        request_id = uuid.uuid4().hex
        ctx = self._agent_ctx(sc, trace_id, request_id, emit, stream)
        await ctx.fire("session", session_id=session.id, trace_id=trace_id)

        # 4. Build the engine (internal or LangGraph) and the initial state, then
        #    RUN the graph. ``thread_id=session.id`` lets LangGraph checkpoint per
        #    session. ``dict(state)`` passes a copy so the engine mutates its own.
        engine = build_engine(ctx, self.settings, checkpointer=self.checkpointer)
        state = make_initial_state(
            sc=sc, question=question, session_id=session.id, trace_id=trace_id,
            request_id=request_id, history=history, summary=session.summary,
        )
        final = await engine.run(dict(state), thread_id=session.id)

        # 5. Pull the results the nodes produced out of the final state.
        answer = final.get("answer", "")
        citations = final.get("citations", []) or []
        route = final.get("route_modules", []) or []
        flags = final.get("output_flags", {}) or {}

        # 6. Persist the assistant's reply (with its citations + routing metadata).
        msg = await self.conversations.append_message(
            sc.org_id, session.id, "assistant", answer, citations=citations,
            meta={"route": route, "flags": flags, "trace_id": trace_id},
        )
        # 7. Maybe roll the summary (if the session got long) and record a
        #    long-term memory observation. The KG write is best-effort: a memory
        #    failure must never break a successful answer, hence the bare except.
        await self._maybe_summarize(sc, session.id)
        try:
            await self.kg.add_observation(sc.org_id, sc.user_id, f"Q: {question}\nA: {answer}")
        except Exception:
            pass

        # 8. Emit the terminal "done" event (streaming) and return the result.
        await ctx.fire("done", answer=answer, citations=citations, session_id=session.id,
                       message_id=msg.id, route=route, flags=flags)
        return TurnResult(
            answer=answer, citations=citations, route_modules=route, output_flags=flags,
            trace_id=trace_id, session_id=session.id, message_id=msg.id,
            route_debug=final.get("route_debug", {}),
        )

    async def stream_turn(
        self, sc: SecurityContext, *, question: str, session_id: str | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Streaming wrapper around run_turn. Pattern: run the turn in a
        background task that pushes events into a queue via its ``emit`` callback,
        while THIS coroutine drains the queue and yields events to the caller
        (the SSE response). The sentinel marks the end."""
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(ev: AgentEvent) -> None:
            await queue.put(ev)

        async def runner() -> None:
            try:
                await self.run_turn(sc, question=question, session_id=session_id, emit=emit, stream=True)
            except Exception as exc:  # noqa: BLE001 - surface as an event, don't crash the stream
                await queue.put(AgentEvent(type="error", data={"message": str(exc)}))
            finally:
                await queue.put(_SENTINEL)   # always signal completion

        task = asyncio.create_task(runner())
        try:
            while True:
                ev = await queue.get()
                if ev is _SENTINEL:
                    break
                yield ev
        finally:
            await task   # ensure the background turn is fully finished/cleaned up

    async def _maybe_summarize(self, sc: SecurityContext, session_id: str) -> None:
        """Roll older turns into a compact summary once the message count crosses
        ``summary_trigger_messages``. This is how the system keeps context bounded
        on long chats (ChatGPT/Claude-style memory) instead of resending the full
        transcript every turn."""
        s = await self.conversations.get_session(sc.org_id, session_id)
        if s and s.message_count >= self.settings.summary_trigger_messages:
            msgs = await self.conversations.get_messages(
                sc.org_id, session_id, limit=self.settings.history_window_messages
            )
            summary = await self.summarizer.summarize(s.summary, msgs)
            await self.conversations.update_session(sc.org_id, session_id, summary=summary)


__all__ = ["Orchestrator", "TurnResult"]
