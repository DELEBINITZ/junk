"""Cross-cutting tools available to every module (not owned by any one of them).

These are registered by the registry alongside module tools and are always in
the router's tool set. `current_date_and_time` anchors relative time queries;
`search_past_conversations` is the cross-session chat recall described in plan
§9.2 (private to the calling user).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from app.config import settings
from app.core.contracts import Tool, ToolContext
from app.core.memory.conversations import get_conversation_store
from app.core.memory.kg import get_kg


def _current_date_and_time(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {"iso": now.isoformat(), "date": now.date().isoformat(), "timezone": "UTC"}


def _search_past_conversations(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    store = get_conversation_store()
    query = str(args.get("query", "")).strip()
    matches = store.search_past(ctx.user, query, since=args.get("since"))
    return {"matches": matches, "count": len(matches)}


def _recall_memory(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    facts = get_kg().search(ctx.org_id, ctx.user.id, str(args.get("query", "")).strip())
    return {"facts": facts, "count": len(facts)}


def build_core_tools() -> list[Tool]:
    tools = [
        Tool(
            name="current_date_and_time",
            description="Return the current UTC date and time. Use to resolve "
            "relative time references like 'last quarter' or 'this year'.",
            input_schema={"type": "object", "properties": {}},
            handler=_current_date_and_time,
            rbac_role="viewer",
        ),
        Tool(
            name="search_past_conversations",
            description="Search the CURRENT user's own previous chat sessions for "
            "relevant prior context. Returns snippets the user can be reminded of.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "since": {"type": "string", "format": "date"},
                },
                "required": ["query"],
            },
            handler=_search_past_conversations,
            rbac_role="viewer",
        ),
    ]
    if os.getenv("KG_PROVIDER", settings.kg_provider).lower() != "none":
        tools.append(
            Tool(
                name="recall_memory",
                description="Recall durable facts about this user/org from the "
                "knowledge graph (entities, prior findings, tracked actors).",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                handler=_recall_memory,
                rbac_role="viewer",
            )
        )
    return tools
