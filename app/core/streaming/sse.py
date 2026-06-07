"""Server-Sent Events: typed event framing for the streaming chat endpoint.

WHAT IS SSE: a dead-simple, one-way "server pushes text to the browser over a
single long-lived HTTP response" protocol. The wire format is just UTF-8 lines:
an optional ``event:`` type line, one or more ``data:`` lines, terminated by a
BLANK line. That blank line is what tells the client "this event is complete" —
get it wrong and events fuse together. We use SSE (not WebSockets) because the
chat stream is strictly server -> client, and SSE needs no extra protocol.

HOW IT FITS: the orchestrator yields typed :class:`AgentEvent`s as a turn runs;
this module is the thin ADAPTER that serializes each one to the SSE wire format.
The typed event vocabulary (status, route, tool, token, citation, done, error)
lets the UI render progressively — show routing, then tool activity, then stream
the answer token-by-token — exactly the ChatGPT/Claude streaming UX.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.core.agent.state import AgentEvent

# The event-type vocabulary — a documented contract the frontend switches on to
# decide how to render each frame (e.g. EV_TOKEN appends text, EV_CITATION adds a
# source chip, EV_DONE closes the turn).
EV_SESSION = "session"
EV_STATUS = "status"
EV_ROUTE = "route"
EV_TOOL = "tool"
EV_THINKING = "thinking"   # human-readable reasoning steps (plan decomposition, reflect verdicts)
EV_TOKEN = "token"
EV_ROLLBACK = "rollback"   # replace all streamed tokens with corrected text (guard redaction/block)
EV_CITATION = "citation"
EV_DONE = "done"
EV_ERROR = "error"


def format_sse(event_type: str, data: dict) -> str:
    """Frame one event in the SSE wire format: an ``event:`` line, a JSON ``data:``
    line, and the mandatory trailing blank line (the ``\\n\\n``) that marks the
    end of the event. The JSON payload is what the client parses."""
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


def comment(text: str) -> str:
    """An SSE comment line (any line starting with ``:``). Clients ignore it; we
    send one as a keep-alive / "stream is open" nudge so proxies don't time out an
    idle connection before the first real event."""
    return f": {text}\n\n"


async def sse_from_events(events: AsyncIterator[AgentEvent]) -> AsyncIterator[str]:
    """Bridge the orchestrator's async event stream to an SSE byte/line stream:
    open with a keep-alive comment, then forward each AgentEvent as a framed SSE
    message. This is the async generator the StreamingResponse iterates."""
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
    "EV_THINKING",
    "EV_TOKEN",
    "EV_CITATION",
    "EV_DONE",
    "EV_ERROR",
]
