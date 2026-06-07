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

from app.core.observability import build_langfuse_handler
from app.core.agent.fast_rag import PARALLEL_PLANNER_NODE_SPECS, build_parallel_planner_graph
from app.core.agent.state import (
    N_ANSWER,
    N_PARALLEL_RETRIEVE,
    N_PLAN_DISPATCH,
    N_REPLAN_GATE,
    AgentContext,
    ChatState,
)


class AgentEngine(Protocol):
    """Both engines satisfy this tiny interface: give me a ``state`` and a
    ``thread_id``, run the graph, hand back the final state."""
    name: str

    async def run(self, state: dict, *, thread_id: str) -> dict: ...


class InternalEngine:
    """The default engine — parallel planner graph (guardrail + plan concurrent,
    wave-based dispatch, reflect loop). Zero external dependencies."""
    name = "internal"

    def __init__(self, ctx: AgentContext) -> None:
        self._compiled = build_parallel_planner_graph(ctx)

    async def run(self, state: dict, *, thread_id: str) -> dict:
        return await self._compiled.run(state)


class LangGraphEngine:
    """Same parallel planner graph mounted on real LangGraph for durable
    checkpointing (resumable runs, human-approval gate)."""

    name = "langgraph"

    def __init__(self, ctx: AgentContext, checkpointer=None) -> None:
        from langgraph.graph import END, START, StateGraph

        sg = StateGraph(ChatState)

        def _wrap(fn):
            async def _node(state):
                return await fn(state, ctx)
            return _node

        for name, fn in PARALLEL_PLANNER_NODE_SPECS:
            sg.add_node(name, _wrap(fn))

        sg.add_edge(START, N_PARALLEL_RETRIEVE)
        sg.add_conditional_edges(
            N_PARALLEL_RETRIEVE,
            lambda s: "blocked" if s.get("blocked") else "ok",
            {"blocked": END, "ok": N_PLAN_DISPATCH},
        )
        sg.add_edge(N_PLAN_DISPATCH, N_ANSWER)
        sg.add_edge(N_ANSWER, N_REPLAN_GATE)
        sg.add_conditional_edges(
            N_REPLAN_GATE,
            lambda s: "replan" if s.get("needs_replan") else "finish",
            {"replan": N_PARALLEL_RETRIEVE, "finish": END},
        )
        self._compiled = sg.compile(checkpointer=checkpointer)

        self._lf_handler = build_langfuse_handler(ctx.settings)
        self._sc = ctx.sc

    async def run(self, state: dict, *, thread_id: str) -> dict:
        cfg = {"configurable": {"thread_id": thread_id}}
        if self._lf_handler is not None:
            cfg["callbacks"] = [self._lf_handler]
            cfg["metadata"] = {
                "langfuse_session_id": thread_id,
                "langfuse_user_id": self._sc.user_id,
                "langfuse_tags": [f"org:{self._sc.org_id}", f"engine:{self.name}"],
            }
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
