"""The orchestrator — the chat service's entry point.

Owns a turn end to end: load history + summary, persist the user message, run the
agent graph (built-in or LangGraph) with a per-request :class:`AgentContext`,
persist the assistant message, update the rolling summary, and (optionally) stream
typed events token-by-token. Sessions, persistence, and streaming live here;
reasoning lives in the graph nodes.
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

_SENTINEL = object()


@dataclass
class TurnResult:
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
        conversations,
        summarizer,
        kg,
        checkpointer=None,
    ) -> None:
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
        return ToolContext(
            org_id=sc.org_id, user_id=sc.user_id, roles=sc.roles,
            trace_id=trace_id, request_id=request_id, deps=self.deps,
        )

    def _agent_ctx(
        self, sc: SecurityContext, trace_id: str, request_id: str,
        emit: EmitFn | None, stream: bool,
    ) -> AgentContext:
        return AgentContext(
            deps=self.deps, sc=sc, tool_ctx=self._tool_ctx(sc, trace_id, request_id),
            mcp=self.mcp, registry=self.registry, input_guard=self.input_guard,
            output_guard=self.output_guard, settings=self.settings, supervisor=self.supervisor,
            emit=emit, stream_tokens=stream,
        )

    async def _ensure_session(self, sc: SecurityContext, session_id: str | None):
        if session_id:
            s = await self.conversations.get_session(sc.org_id, session_id)
            if not s:
                raise NotFound(f"session '{session_id}' not found", details={"session_id": session_id})
            return s
        return await self.conversations.create_session(sc.org_id, sc.user_id)

    async def preview_route(self, sc: SecurityContext, question: str) -> RouteResult:
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
        session = await self._ensure_session(sc, session_id)
        prior = await self.conversations.get_messages(
            sc.org_id, session.id, limit=self.settings.history_window_messages
        )
        history = [{"role": m.role, "content": m.content} for m in prior]
        await self.conversations.append_message(sc.org_id, session.id, "user", question)

        trace_id = uuid.uuid4().hex
        request_id = uuid.uuid4().hex
        ctx = self._agent_ctx(sc, trace_id, request_id, emit, stream)
        await ctx.fire("session", session_id=session.id, trace_id=trace_id)

        engine = build_engine(ctx, self.settings, checkpointer=self.checkpointer)
        state = make_initial_state(
            sc=sc, question=question, session_id=session.id, trace_id=trace_id,
            request_id=request_id, history=history, summary=session.summary,
        )
        final = await engine.run(dict(state), thread_id=session.id)

        answer = final.get("answer", "")
        citations = final.get("citations", []) or []
        route = final.get("route_modules", []) or []
        flags = final.get("output_flags", {}) or {}

        msg = await self.conversations.append_message(
            sc.org_id, session.id, "assistant", answer, citations=citations,
            meta={"route": route, "flags": flags, "trace_id": trace_id},
        )
        await self._maybe_summarize(sc, session.id)
        try:
            await self.kg.add_observation(sc.org_id, sc.user_id, f"Q: {question}\nA: {answer}")
        except Exception:
            pass

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
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(ev: AgentEvent) -> None:
            await queue.put(ev)

        async def runner() -> None:
            try:
                await self.run_turn(sc, question=question, session_id=session_id, emit=emit, stream=True)
            except Exception as exc:  # noqa: BLE001 - surface as an event
                await queue.put(AgentEvent(type="error", data={"message": str(exc)}))
            finally:
                await queue.put(_SENTINEL)

        task = asyncio.create_task(runner())
        try:
            while True:
                ev = await queue.get()
                if ev is _SENTINEL:
                    break
                yield ev
        finally:
            await task

    async def _maybe_summarize(self, sc: SecurityContext, session_id: str) -> None:
        s = await self.conversations.get_session(sc.org_id, session_id)
        if s and s.message_count >= self.settings.summary_trigger_messages:
            msgs = await self.conversations.get_messages(
                sc.org_id, session_id, limit=self.settings.history_window_messages
            )
            summary = await self.summarizer.summarize(s.summary, msgs)
            await self.conversations.update_session(sc.org_id, session_id, summary=summary)


__all__ = ["Orchestrator", "TurnResult"]
