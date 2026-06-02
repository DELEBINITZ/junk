"""Engine factory: the built-in engine and the REAL LangGraph engine.

This file is the bridge to LangGraph the library. The trick that makes it work:
both engines run the IDENTICAL node set (``nodes.NODE_SPECS``) with the SAME
edges. So you can develop and test with zero dependencies on the built-in engine
(graph.py), then flip ``agent_engine=langgraph`` to run the exact same agent on
real LangGraph — gaining its big feature, durable CHECKPOINTING.

What is checkpointing / why care? LangGraph can persist the state after each node
to a "checkpointer" (memory or Postgres). That makes a run RESUMABLE: it can stop
mid-graph (e.g. waiting on a human to approve a side-effecting action) and later
continue from exactly where it paused, keyed by a ``thread_id``. That is the
foundation for the human-approval gate. Our built-in engine has no persistence;
LangGraph gives it to us for free once we mirror the nodes onto it.
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
    """Both engines satisfy this tiny interface, so the orchestrator doesn't care
    which one it's holding: give me a ``state`` and a ``thread_id``, run the
    graph, hand back the final state."""
    name: str

    async def run(self, state: dict, *, thread_id: str) -> dict: ...


class InternalEngine:
    """The default, zero-dependency engine — just wraps our built-in graph
    (graph.py via nodes.build_report_graph). ``thread_id`` is accepted for
    interface parity but unused: the built-in engine keeps no cross-run state."""
    name = "internal"

    def __init__(self, ctx: AgentContext) -> None:
        self._compiled = build_report_graph(ctx)

    async def run(self, state: dict, *, thread_id: str) -> dict:
        return await self._compiled.run(state)


class LangGraphEngine:
    """The same five nodes mounted onto real ``langgraph.StateGraph``, plus a
    checkpointer. Compare this wiring to nodes.build_report_graph — it is the
    SAME shape (entry -> conditional after input guard -> linear to END), just
    expressed in LangGraph's API (START/END come from the library here)."""

    name = "langgraph"

    def __init__(self, ctx: AgentContext, checkpointer=None) -> None:
        # Imported lazily so the langgraph package is only needed when this engine
        # is actually selected (keeps the default install dependency-free).
        from langgraph.graph import END, START, StateGraph

        # LangGraph wants the state SCHEMA (our ChatState TypedDict) up front.
        sg = StateGraph(ChatState)

        # Our node functions take ``(state, ctx)``; LangGraph calls nodes with just
        # ``(state)``. This wrapper closes over ``ctx`` to bridge the two — exactly
        # the same closure trick the built-in engine uses.
        def _wrap(fn):
            async def _node(state):
                return await fn(state, ctx)
            return _node

        # Register the IDENTICAL node set (single source of truth = NODE_SPECS).
        for name, fn in NODE_SPECS:
            sg.add_node(name, _wrap(fn))
        sg.add_edge(START, N_INPUT_GUARD)
        # Same single branch as the built-in graph: blocked -> END, else -> route.
        sg.add_conditional_edges(
            N_INPUT_GUARD,
            lambda s: "blocked" if s.get("blocked") else "ok",
            {"blocked": END, "ok": N_ROUTE},
        )
        sg.add_edge(N_ROUTE, N_GATHER)
        sg.add_edge(N_GATHER, N_ANSWER)
        sg.add_edge(N_ANSWER, N_OUTPUT_GUARD)
        sg.add_edge(N_OUTPUT_GUARD, END)
        # ``compile(checkpointer=...)`` is what enables resumable, persisted runs.
        self._compiled = sg.compile(checkpointer=checkpointer)

    async def run(self, state: dict, *, thread_id: str) -> dict:
        # ``thread_id`` keys the checkpoint — reuse the same id (we use the chat
        # session id) and LangGraph can associate/resume the run.
        cfg = {"configurable": {"thread_id": thread_id}}
        return await self._compiled.ainvoke(state, cfg)


def build_checkpointer(settings):
    """Create the LangGraph checkpointer once and reuse it. Prefer durable
    Postgres persistence when a database is configured; otherwise fall back to an
    in-memory saver (resumable within the process only). Any failure wiring
    Postgres degrades to memory rather than crashing boot."""
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
    """Choose the engine from config. This is the ONE place the two engines
    diverge; everything upstream (orchestrator, nodes) is engine-agnostic."""
    if settings.agent_engine == "langgraph":
        return LangGraphEngine(ctx, checkpointer=checkpointer)
    return InternalEngine(ctx)


__all__ = ["AgentEngine", "InternalEngine", "LangGraphEngine", "build_engine", "build_checkpointer"]
