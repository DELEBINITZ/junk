"""The manifest-driven supervisor.

`run_query` runs the LangGraph-shaped graph (nodes.py) and returns an AgentTurn.
`stream_events` runs the same guardrail->route->retrieve flow but streams the
answer token-by-token (real tokens from a streaming LLM, else chunked) — sharing
answer composition with the graph via answering.py.

Deps are config-driven and default to the local path: keyword router (or LLM
planner when ROUTER_MODE=llm), SGLang lanes, Qdrant retrieval, model guardrails,
and Langfuse tracing all swap in via settings without changing this file.
See plan §6, §11, §12.
"""

from __future__ import annotations

import logging
from typing import Iterator
from uuid import uuid4

from app.config import settings
from app.core.agent.answering import (
    NO_ACCESS_MESSAGE,
    REFUSAL_MESSAGE,
    compose_grounded,
    stream_answer,
)
from app.core.agent.nodes import AgentContext, build_report_graph
from app.core.agent.state import AgentTurn
from app.core.contracts import ToolContext, ToolResult, is_error
from app.core.guardrails.pipeline import InputGuardrailPipeline, OutputGuardrailPipeline
from app.core.llm.lanes import Lane, LaneRouter
from app.core.memory.conversations import get_conversation_store
from app.core.observability.metrics import metrics
from app.core.observability.tracing import get_tracer
from app.core.registry import CapabilityRegistry, get_registry
from app.core.router import LLMPlanner, Router
from app.db.repository import DataStore
from app.domain import User
from app.llm.client import estimate_tokens


logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        user: User,
        store: DataStore,
        registry: CapabilityRegistry | None = None,
        conversations=None,
        max_top_k: int = 5,
    ):
        self.user = user
        self.store = store
        self.registry = registry or get_registry()
        self.conv = conversations or get_conversation_store()
        self.lanes = LaneRouter()
        self.router = Router(self.registry, planner=self._build_planner())
        self.input_guard = InputGuardrailPipeline()
        self.output_guard = OutputGuardrailPipeline()
        self.max_top_k = max_top_k

    def _build_planner(self) -> LLMPlanner | None:
        if settings.router_mode.lower() == "llm":
            return LLMPlanner(self.lanes.client_for(Lane.FAST))
        return None

    def _agent_context(self, trace_id: str) -> AgentContext:
        return AgentContext(
            user=self.user, store=self.store, registry=self.registry, router=self.router,
            conversations=self.conv, input_guard=self.input_guard, output_guard=self.output_guard,
            lanes=self.lanes, trace_id=trace_id, max_top_k=self.max_top_k,
        )

    def _tool_context(self, trace_id: str) -> ToolContext:
        return ToolContext(org_id=self.user.organization_id, user=self.user, trace_id=trace_id, store=self.store)

    # ---- non-streaming -----------------------------------------------------------
    def run_query(self, message: str, session_id: str | None = None) -> AgentTurn:
        trace_id = str(uuid4())
        metrics.incr("orchestrator.run_query")
        with get_tracer().span("agent.query", org_id=self.user.organization_id, trace_id=trace_id):
            ctx = self._agent_context(trace_id)
            state = {
                "user_message": message, "session_id": session_id,
                "tool_calls": [], "retrieved": [], "citations": [], "route_module_ids": [],
            }
            if settings.agent_engine.lower() == "langgraph":
                from app.core.agent.langgraph_engine import run_langgraph

                final = run_langgraph(ctx, state)
            else:
                final = build_report_graph(ctx).run(state)
        status = "refused" if final.get("refused") else "ok"
        return AgentTurn(
            status=status,
            answer=final.get("final_answer", ""),
            citations=final.get("citations", []),
            module_ids=final.get("route_module_ids", []),
            tool_calls=final.get("tool_calls", []),
            retrieved=final.get("retrieved", []),
            lane=final.get("lane", "standard"),
            trace_id=trace_id,
            session_id=final.get("session_id", "") or "",
            tokens={"input": estimate_tokens(message), "output": estimate_tokens(final.get("final_answer", ""))},
        )

    # ---- streaming ---------------------------------------------------------------
    def stream_events(self, message: str, session_id: str | None = None) -> Iterator[tuple[str, dict]]:
        """Yield semantic events: ("status"|"tool_call"|"tool_result"|"token"|
        "citation"|"done", payload). The SSE layer maps these to typed events."""

        trace_id = str(uuid4())
        metrics.incr("orchestrator.stream")
        yield "status", {"phase": "thinking"}

        verdict = self.input_guard.run(message)
        redacted = verdict.redacted_text or message
        user_message = self.conv.append(self.user, session_id, "user", redacted)
        session_id = user_message.session_id

        if not verdict.allowed:
            for word in REFUSAL_MESSAGE.split(" "):
                yield "token", {"text": word + " "}
            yield "done", {"status": "refused", "session_id": session_id, "trace_id": trace_id}
            return

        decision = self.router.route(redacted, self.user)
        yield "status", {"phase": "routed", "modules": decision.module_ids, "lane": decision.lane}

        citations: list[str] = []
        if "search_contracts" in decision.tool_names:
            tool = self.registry.tool("search_contracts")
            result = tool.run({"query": redacted, "top_k": self.max_top_k}, self._tool_context(trace_id))
            ok = not is_error(result)
            yield "tool_call", {"tool": "search_contracts"}
            yield "tool_result", {"tool": "search_contracts", "ok": ok}
            retrieved = result.data.get("matches", []) if isinstance(result, ToolResult) else []
            grounded, citations = compose_grounded(retrieved, self.max_top_k)
            client = self.lanes.client_for(decision.lane)
            accumulated = ""
            for chunk in stream_answer(client, redacted, grounded):
                accumulated += chunk
                yield "token", {"text": chunk}
            answer = accumulated.strip() or grounded
        else:
            answer = NO_ACCESS_MESSAGE
            for word in answer.split(" "):
                yield "token", {"text": word + " "}

        verdict_out = self.output_guard.run(answer, citations)
        answer = verdict_out.redacted_text or answer
        self.conv.append(self.user, session_id, "assistant", answer, citations)

        for index, marker in enumerate(citations, start=1):
            yield "citation", {"id": index, "marker": marker}
        yield "done", {
            "status": "ok", "session_id": session_id, "trace_id": trace_id,
            "tokens": {"input": estimate_tokens(message), "output": estimate_tokens(answer)},
        }
