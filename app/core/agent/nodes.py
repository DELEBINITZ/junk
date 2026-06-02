"""The agent graph's nodes (plan §6).

Pipeline: load_session_state -> input_guardrail -> persist_user ->
(refusal | plan_route) -> retrieve -> answer -> output_guardrail ->
persist_assistant -> END.

Nodes are closures over an AgentContext. Behavior matches the original
orchestrator (refusal on injection, no-access for viewers, grounded cited
answers); answer composition is shared with the streaming path via answering.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.agent.answering import (
    NO_ACCESS_MESSAGE,
    REFUSAL_MESSAGE,
    compose_grounded,
    finalize_answer,
)
from app.core.agent.graph import END, Graph
from app.core.contracts import ToolContext, ToolResult, is_error
from app.core.guardrails.pipeline import InputGuardrailPipeline, OutputGuardrailPipeline
from app.core.llm.lanes import LaneRouter
from app.core.registry import CapabilityRegistry
from app.core.router import Router
from app.db.repository import DataStore
from app.domain import User


@dataclass
class AgentContext:
    user: User
    store: DataStore
    registry: CapabilityRegistry
    router: Router
    conversations: Any
    input_guard: InputGuardrailPipeline
    output_guard: OutputGuardrailPipeline
    lanes: LaneRouter
    trace_id: str
    max_top_k: int = 5

    def tool_context(self) -> ToolContext:
        return ToolContext(
            org_id=self.user.organization_id, user=self.user,
            trace_id=self.trace_id, store=self.store,
        )


def build_report_graph(ctx: AgentContext) -> Graph:
    def load_session_state(state):
        session_id = state.get("session_id")
        history = ctx.conversations.recent_history(ctx.user, session_id, k=6) if session_id else []
        return {"history": [m.role for m in history]}

    def input_guardrail(state):
        verdict = ctx.input_guard.run(state["user_message"])
        update = {"rewritten_query": verdict.redacted_text or state["user_message"]}
        if not verdict.allowed:
            update.update(refused=True, final_answer=REFUSAL_MESSAGE,
                          guardrail_violations=[verdict.category])
        else:
            update["refused"] = False
        return update

    def persist_user(state):
        message = ctx.conversations.append(ctx.user, state.get("session_id"), "user", state["rewritten_query"])
        return {"session_id": message.session_id}

    def plan_route(state):
        decision = ctx.router.route(state["rewritten_query"], ctx.user)
        return {
            "route_module_ids": decision.module_ids,
            "route_tool_names": decision.tool_names,
            "lane": decision.lane,
        }

    def retrieve(state):
        if "search_contracts" not in state.get("route_tool_names", []):
            return {"no_access": True, "tool_calls": [], "retrieved": []}
        tool = ctx.registry.tool("search_contracts")
        result = tool.run({"query": state["rewritten_query"], "top_k": ctx.max_top_k}, ctx.tool_context())
        call = {"tool": "search_contracts", "ok": not is_error(result), "code": getattr(result, "code", 200)}
        retrieved = result.data.get("matches", []) if isinstance(result, ToolResult) else []
        return {"no_access": False, "tool_calls": [call], "retrieved": retrieved,
                "retrieve_error": None if isinstance(result, ToolResult) else result.message}

    def answer(state):
        if state.get("no_access"):
            return {"final_answer": NO_ACCESS_MESSAGE, "citations": []}
        if state.get("retrieve_error"):
            return {"final_answer": f"Could not retrieve reports: {state['retrieve_error']}", "citations": []}
        grounded, citations = compose_grounded(state.get("retrieved", []), ctx.max_top_k)
        client = ctx.lanes.client_for(state.get("lane", "standard"))
        text = finalize_answer(client, state["rewritten_query"], grounded)
        return {"final_answer": text, "citations": citations}

    def output_guardrail(state):
        verdict = ctx.output_guard.run(state.get("final_answer", ""), state.get("citations", []))
        return {"final_answer": verdict.redacted_text or state.get("final_answer", "")}

    def persist_assistant(state):
        ctx.conversations.append(ctx.user, state["session_id"], "assistant",
                                 state.get("final_answer", ""), state.get("citations", []))
        return {}

    def refusal(state):
        return {"final_answer": state.get("final_answer") or REFUSAL_MESSAGE}

    return Graph(
        entry="load_session_state",
        nodes={
            "load_session_state": load_session_state,
            "input_guardrail": input_guardrail,
            "persist_user": persist_user,
            "plan_route": plan_route,
            "retrieve": retrieve,
            "answer": answer,
            "output_guardrail": output_guardrail,
            "persist_assistant": persist_assistant,
            "refusal": refusal,
        },
        edges={
            "load_session_state": "input_guardrail",
            "input_guardrail": "persist_user",
            "persist_user": lambda s: "refusal" if s.get("refused") else "plan_route",
            "refusal": END,
            "plan_route": "retrieve",
            "retrieve": "answer",
            "answer": "output_guardrail",
            "output_guardrail": "persist_assistant",
            "persist_assistant": END,
        },
    )
