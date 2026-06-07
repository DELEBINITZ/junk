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
from app.core.agent.nodes import (
    NODE_SPECS,
    PLANNER_NODE_SPECS,
    build_planner_graph,
    build_report_graph,
)
from app.core.agent.state import (
    N_ANSWER,
    N_GATHER,
    N_INPUT_GUARD,
    N_OUTPUT_GUARD,
    N_PLAN,
    N_PLAN_DISPATCH,
    N_REPLAN_GATE,
    N_ROUTE,
    N_TRIAGE,
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
    (graph.py). Picks the heuristic graph or the planner graph from
    ``orchestrator_mode``. ``thread_id`` is accepted for interface parity but
    unused: the built-in engine keeps no cross-run state."""
    name = "internal"

    def __init__(self, ctx: AgentContext) -> None:
        if getattr(ctx.settings, "orchestrator_mode", "heuristic") == "planner":
            self._compiled = build_planner_graph(ctx)
        else:
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

        planner_mode = getattr(ctx.settings, "orchestrator_mode", "heuristic") == "planner"
        # Register the IDENTICAL node set the built-in engine uses (single source of
        # truth), choosing the heuristic or planner node list by mode.
        specs = PLANNER_NODE_SPECS if planner_mode else NODE_SPECS
        for name, fn in specs:
            sg.add_node(name, _wrap(fn))
        sg.add_edge(START, N_INPUT_GUARD)
        # blocked -> END; else triage for small-talk (both modes).
        sg.add_conditional_edges(
            N_INPUT_GUARD,
            lambda s: "blocked" if s.get("blocked") else "ok",
            {"blocked": END, "ok": N_TRIAGE},
        )
        if planner_mode:
            # triage always passes through to planner — LLM handles all messages.
            sg.add_edge(N_TRIAGE, N_PLAN)
            sg.add_edge(N_PLAN, N_PLAN_DISPATCH)
            sg.add_edge(N_PLAN_DISPATCH, N_ANSWER)
            sg.add_edge(N_ANSWER, N_REPLAN_GATE)
            sg.add_conditional_edges(
                N_REPLAN_GATE,
                lambda s: "replan" if s.get("needs_replan") else "finish",
                {"replan": N_PLAN, "finish": N_OUTPUT_GUARD},
            )
        else:
            # triage always passes through to routing — LLM handles all messages.
            sg.add_edge(N_TRIAGE, N_ROUTE)
            sg.add_edge(N_ROUTE, N_GATHER)
            sg.add_edge(N_GATHER, N_ANSWER)
            sg.add_edge(N_ANSWER, N_OUTPUT_GUARD)
        sg.add_edge(N_OUTPUT_GUARD, END)
        # ``compile(checkpointer=...)`` is what enables resumable, persisted runs.
        self._compiled = sg.compile(checkpointer=checkpointer)

        # Auto-trace seam: a Langfuse callback handler (or None when tracing is
        # off) that LangGraph invokes for every node/LLM step, building the full
        # per-node trace tree in the dashboard — the LangSmith-style view, self-
        # hosted. Built ONCE here because the keys are static; the per-run session
        # /user metadata is attached in run(). ``sc`` is the acting identity, used
        # to group traces by chat session, user, and tenant.
        self._lf_handler = build_langfuse_handler(ctx.settings)
        self._sc = ctx.sc

    async def run(self, state: dict, *, thread_id: str) -> dict:
        # ``thread_id`` keys the checkpoint — reuse the same id (we use the chat
        # session id) and LangGraph can associate/resume the run.
        cfg = {"configurable": {"thread_id": thread_id}}
        # When Langfuse is configured, attach the auto-tracing handler and tag the
        # run so the dashboard groups this chat session's turns together
        # (langfuse_session_id) under the acting user/tenant. The langfuse_* keys
        # are read by the handler; they are inert when no handler is attached.
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
