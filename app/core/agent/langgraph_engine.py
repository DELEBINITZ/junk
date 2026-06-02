"""Real LangGraph engine (plan §6) — opt-in via AGENT_ENGINE=langgraph.

Reuses the SAME node functions as the built-in graph (nodes.py) and mirrors the
edges into a `langgraph.StateGraph`, compiled with a checkpointer (PostgresSaver
when STORE_BACKEND=postgres, else MemorySaver) keyed by session_id — giving
durable, resumable runs. `langgraph` is imported lazily so the default built-in
engine needs no dependency.
"""

from __future__ import annotations

from app.config import settings
from app.core.agent.nodes import AgentContext, build_report_graph


def _checkpointer():
    if settings.store_backend.lower() == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            return PostgresSaver.from_conn_string(settings.database_url)
        except Exception:
            pass
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


def build_langgraph(ctx: AgentContext):
    """Build + compile the LangGraph equivalent of the built-in report graph."""

    from langgraph.graph import END, START, StateGraph

    nodes = build_report_graph(ctx).nodes  # reuse the exact node logic
    builder = StateGraph(dict)
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "load_session_state")
    builder.add_edge("load_session_state", "input_guardrail")
    builder.add_edge("input_guardrail", "persist_user")
    builder.add_conditional_edges(
        "persist_user",
        lambda s: "refusal" if s.get("refused") else "plan_route",
        {"refusal": "refusal", "plan_route": "plan_route"},
    )
    builder.add_edge("refusal", END)
    builder.add_edge("plan_route", "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", "output_guardrail")
    builder.add_edge("output_guardrail", "persist_assistant")
    builder.add_edge("persist_assistant", END)

    return builder.compile(checkpointer=_checkpointer())


def run_langgraph(ctx: AgentContext, state: dict) -> dict:
    graph = build_langgraph(ctx)
    thread_id = state.get("session_id") or "default"
    return graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
