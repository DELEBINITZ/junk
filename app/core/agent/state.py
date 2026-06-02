"""Orchestrator state + the structured result of one turn.

ChatState mirrors the LangGraph state schema in plan Appendix B so the migration
to a compiled graph is mechanical. AgentTurn is what the API returns and what the
SSE streamer serializes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class ChatState(TypedDict, total=False):
    session_id: str
    user_id: str
    org_id: str
    trace_id: str
    user_message: str
    rewritten_query: str
    route_module_ids: list[str]
    route_tool_names: list[str]
    lane: str
    tool_calls: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    citations: list[str]
    final_answer: str
    guardrail_violations: list[str]
    refused: bool
    error: str | None


@dataclass
class AgentTurn:
    status: str  # "ok" | "refused" | "error"
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    module_ids: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    lane: str = "standard"
    trace_id: str = ""
    session_id: str = ""
    tokens: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "citations": self.citations,
            "module_ids": self.module_ids,
            "tool_calls": self.tool_calls,
            "retrieved": self.retrieved,
            "lane": self.lane,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "tokens": self.tokens,
            "error": self.error,
        }
