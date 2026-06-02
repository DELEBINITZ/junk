"""Engine factory: the built-in engine and the real LangGraph engine.

Both run the IDENTICAL node set (``nodes.NODE_SPECS``) and the same edges, so
behavior matches; the LangGraph engine adds durable checkpointing (MemorySaver,
or PostgresSaver when Postgres is configured) which is what makes runs resumable
for the future human-approval gate. Selected by ``agent_engine``.
"""

from __future__ import annotations

from typing import Protocol

from app.core.agent.nodes import NODE_SPECS, build_report_graph
from app.core.agent.state import (
    N_ANSWER,
    N_GATHER,
    N_INPUT_GUARD,
    N_OUTPUT_GUARD,
    N_ROUTE,
    AgentContext,
    ChatState,
)


class AgentEngine(Protocol):
    name: str

    async def run(self, state: dict, *, thread_id: str) -> dict: ...


class InternalEngine:
    name = "internal"

    def __init__(self, ctx: AgentContext) -> None:
        self._compiled = build_report_graph(ctx)

    async def run(self, state: dict, *, thread_id: str) -> dict:
        return await self._compiled.run(state)


class LangGraphEngine:
    """Mirror of the node set onto ``langgraph.StateGraph`` with a checkpointer."""

    name = "langgraph"

    def __init__(self, ctx: AgentContext, checkpointer=None) -> None:
        from langgraph.graph import END, START, StateGraph

        sg = StateGraph(ChatState)

        def _wrap(fn):
            async def _node(state):
                return await fn(state, ctx)
            return _node

        for name, fn in NODE_SPECS:
            sg.add_node(name, _wrap(fn))
        sg.add_edge(START, N_INPUT_GUARD)
        sg.add_conditional_edges(
            N_INPUT_GUARD,
            lambda s: "blocked" if s.get("blocked") else "ok",
            {"blocked": END, "ok": N_ROUTE},
        )
        sg.add_edge(N_ROUTE, N_GATHER)
        sg.add_edge(N_GATHER, N_ANSWER)
        sg.add_edge(N_ANSWER, N_OUTPUT_GUARD)
        sg.add_edge(N_OUTPUT_GUARD, END)
        self._compiled = sg.compile(checkpointer=checkpointer)

    async def run(self, state: dict, *, thread_id: str) -> dict:
        cfg = {"configurable": {"thread_id": thread_id}}
        return await self._compiled.ainvoke(state, cfg)


def build_checkpointer(settings):
    """Shared checkpointer for the LangGraph engine (created once, reused)."""
    from langgraph.checkpoint.memory import MemorySaver

    if settings.store_backend == "postgres" and settings.database_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # optional pkg

            saver = PostgresSaver.from_conn_string(settings.database_url)
            saver.setup()
            return saver
        except Exception:
            return MemorySaver()
    return MemorySaver()


def build_engine(ctx: AgentContext, settings, *, checkpointer=None) -> AgentEngine:
    if settings.agent_engine == "langgraph":
        return LangGraphEngine(ctx, checkpointer=checkpointer)
    return InternalEngine(ctx)


__all__ = ["AgentEngine", "InternalEngine", "LangGraphEngine", "build_engine", "build_checkpointer"]
