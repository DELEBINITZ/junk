"""SSE typed events (plan §11).

`stream_turn` maps the orchestrator's semantic event stream 1:1 onto the SSE wire
format (status / tool_call / tool_result / token / citation / done). Token events
now carry real per-token output when a streaming LLM is configured; otherwise the
grounded summary is chunked — same wire format either way.
"""

from __future__ import annotations

import json
from typing import Iterator

from app.core.agent.orchestrator import Orchestrator


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_turn(orchestrator: Orchestrator, message: str, session_id: str | None = None) -> Iterator[str]:
    for kind, payload in orchestrator.stream_events(message, session_id=session_id):
        yield sse_event(kind, payload)
