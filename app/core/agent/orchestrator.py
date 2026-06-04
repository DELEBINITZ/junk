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
        conversations,   # the session/message store (Postgres + RLS)
        summarizer,      # rolls long histories into a short summary
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
        # Cross-session memory: pull relevant snippets from the user's OTHER conversations
        # so follow-ups referencing past chats have continuity (best-effort).
        recalled = await self._recall(sc, session.id, question)
        state = make_initial_state(
            sc=sc, question=question, session_id=session.id, trace_id=trace_id,
            request_id=request_id, history=history, summary=session.summary, recalled=recalled,
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
        # 7. Maybe roll the summary (if the session got long) so a long conversation
        #    stays bounded for the next turn.
        await self._maybe_summarize(sc, session.id)

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
            # The generator is closed either because the turn finished (task already
            # done -> cancel is a no-op) OR because the CLIENT DISCONNECTED (Starlette
            # throws GeneratorExit in here). In the disconnect case we must CANCEL the
            # background turn, not await it: a gone client must never keep the full
            # agent — and its expensive 72B calls — running. Cancellation propagates
            # into the in-flight LLM stream, so vLLM aborts generation too; DB work
            # unwinds via its transaction context managers (rollback). CancelledError
            # is expected here and swallowed.
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _recall(self, sc: SecurityContext, session_id: str, question: str) -> list[dict]:
        """CROSS-SESSION MEMORY: return a few relevant snippets from the user's OTHER
        conversations (not the current session), so a follow-up that references an earlier
        chat has continuity. The store scopes the search to (org_id, user_id) — recall can
        never cross tenant or user lines. Best-effort: a failure returns nothing rather
        than breaking the turn. Injected as background context only (not citable evidence)."""
        if not getattr(self.settings, "cross_session_recall", False):
            return []
        k = getattr(self.settings, "cross_session_recall_k", 3)
        try:
            hits = await self.conversations.search_messages(sc.org_id, sc.user_id, question, limit=k * 4)
        except Exception:  # noqa: BLE001 - recall must never break a turn
            return []
        out: list[dict] = []
        for m in hits:
            if m.session_id == session_id or not (m.content or "").strip():
                continue                                  # skip the current chat + empties
            out.append({"role": m.role, "content": m.content.strip()[:300], "session_id": m.session_id})
            if len(out) >= k:
                break
        return out

    async def _maybe_summarize(self, sc: SecurityContext, session_id: str) -> None:
        """Fold the turns that just fell OUT of the live history window into the
        rolling summary — ChatGPT/Claude-style compaction that loses nothing.

        The live prompt shows only the newest ``history_window_messages`` raw; every
        message older than that must be carried by ``summary`` instead. ``summarized_upto``
        is the watermark of how many oldest messages are already summarized, so each turn
        we compact ONLY the slice that is (a) now evicted from the window and (b) not yet
        summarized — never the recent turns (the old bug re-summarized the recent ones
        while the genuinely old turns were dropped from context forever)."""
        s = await self.conversations.get_session(sc.org_id, session_id)
        if not s:
            return
        # Everything older than the newest ``window`` messages belongs in the summary.
        target = s.message_count - self.settings.history_window_messages
        if target <= s.summarized_upto:
            return                              # nothing newly evicted -> skip (watermark guard)
        evicted = await self.conversations.get_messages(
            sc.org_id, session_id, offset=s.summarized_upto, limit=target - s.summarized_upto
        )
        if not evicted:
            return
        summary = await self.summarizer.summarize(s.summary, evicted)
        await self.conversations.update_session(
            sc.org_id, session_id, summary=summary, summarized_upto=target
        )


__all__ = ["Orchestrator", "TurnResult"]
