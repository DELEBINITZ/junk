"""Server-Sent Events: typed event framing for the streaming chat endpoint.

The orchestrator yields :class:`AgentEvent`s; this turns them into the SSE wire
format. Typed events (status, route, tool, token, citation, done, error) let the
UI render progressively — exactly the ChatGPT/Claude streaming UX.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.core.agent.state import AgentEvent

# event types (documented contract for clients)
EV_SESSION = "session"
EV_STATUS = "status"
EV_ROUTE = "route"
EV_TOOL = "tool"
EV_TOKEN = "token"
EV_CITATION = "citation"
EV_DONE = "done"
EV_ERROR = "error"


def format_sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


def comment(text: str) -> str:
    """SSE comment line (used as keep-alive)."""
    return f": {text}\n\n"


async def sse_from_events(events: AsyncIterator[AgentEvent]) -> AsyncIterator[str]:
    yield comment("stream open")
    async for ev in events:
        yield format_sse(ev.type, ev.data)


__all__ = [
    "format_sse",
    "comment",
    "sse_from_events",
    "EV_SESSION",
    "EV_STATUS",
    "EV_ROUTE",
    "EV_TOOL",
    "EV_TOKEN",
    "EV_CITATION",
    "EV_DONE",
    "EV_ERROR",
]
