"""Build the trusted per-call ToolContext from the authenticated user.

org_id always comes from the verified User (derived from the JWT), never from
request/tool arguments — the single rule that keeps tenancy safe (plan §8.2).
"""

from __future__ import annotations

from uuid import uuid4

from app.core.contracts import ToolContext
from app.domain import User


def build_tool_context(user: User, store, trace_id: str | None = None, kg=None) -> ToolContext:
    return ToolContext(
        org_id=user.organization_id,
        user=user,
        trace_id=trace_id or str(uuid4()),
        store=store,
        kg=kg,
    )
