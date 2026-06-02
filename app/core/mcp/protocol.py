"""Minimal MCP wire shapes — the OVER-THE-NETWORK form of the tool boundary.

The in-process client (inprocess.py) calls tools as plain Python. But once a
module is promoted to its own service, the SAME two operations have to travel
over HTTP. MCP defines them as JSON-RPC 2.0 methods:

  * ``tools/list`` — "what tools do you have?"
  * ``tools/call`` — "run this named tool with these arguments."

JSON-RPC is a tiny convention for remote calls: the caller sends
``{jsonrpc, id, method, params}`` and gets back ``{jsonrpc, id, result|error}``,
where ``id`` correlates the reply to its request. This file is JUST those
envelopes plus translators between our internal ``ToolOutcome`` (ToolResult /
ToolError) and MCP's result format.

WHY IT MATTERS: because the wire shape is identical for both transports,
promoting a module from in-process to a standalone ``easm-mcp`` server is a
TRANSPORT swap, not a contract change — the agent code is untouched. The remote
client (remote.py) speaks this; the standalone server (server.py) answers it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.contracts import ToolError, ToolOutcome, ToolResult

# The two MCP method names, as constants so client and server can never drift
# on a typo'd string.
METHOD_TOOLS_LIST = "tools/list"
METHOD_TOOLS_CALL = "tools/call"


class JSONRPCRequest(BaseModel):
    """A JSON-RPC 2.0 request envelope. ``method`` names the operation,
    ``params`` carries its arguments, and ``id`` is an opaque correlation token
    echoed back in the response so a caller can match replies to requests."""

    jsonrpc: str = "2.0"
    id: Any | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JSONRPCResponse(BaseModel):
    """A JSON-RPC 2.0 response envelope. EXACTLY ONE of ``result`` (success) or
    ``error`` (failure) is set, and ``id`` matches the request that produced it."""

    jsonrpc: str = "2.0"
    id: Any | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


def outcome_to_mcp(outcome: ToolOutcome) -> dict[str, Any]:
    """SERVER side: pack our internal ToolResult/ToolError into MCP's result
    shape. MCP expects both a machine-readable ``structuredContent`` (the full
    model, so the caller can rebuild the typed object) and a human-readable
    ``content`` text block, with ``isError`` flagging failures. We emit both so a
    generic MCP client gets useful text while ours recovers the exact object."""
    if isinstance(outcome, ToolResult):
        return {
            "isError": False,
            "structuredContent": outcome.model_dump(),
            "content": [{"type": "text", "text": _summarize_result(outcome)}],
        }
    err: ToolError = outcome
    return {
        "isError": True,
        "structuredContent": err.model_dump(),
        "content": [{"type": "text", "text": f"{err.code}: {err.message}"}],
    }


def mcp_to_outcome(envelope: dict[str, Any]) -> ToolOutcome:
    """CLIENT side: the inverse — rebuild a ToolResult/ToolError from the MCP
    envelope returned by a remote server. Reads ``structuredContent`` to
    reconstruct the typed object; ``isError`` decides which class. This is what
    lets a REMOTE tool call return the same outcome type as a local one, so the
    agent can't tell in-process from remote."""
    sc = envelope.get("structuredContent") or {}
    if envelope.get("isError"):
        # Backfill code/message defaults so a sparse error envelope still yields a valid ToolError.
        return ToolError(**{**{"code": "remote_error", "message": "remote tool error"}, **sc})
    return ToolResult(**sc) if sc else ToolResult()


def _summarize_result(r: ToolResult) -> str:
    """A one-line human gist of a successful result (citation count + which data
    fields were returned) — this fills MCP's ``content`` text block."""
    keys = ", ".join(r.data.keys())
    return f"ok ({len(r.citations)} citations){'; fields: ' + keys if keys else ''}"


__all__ = [
    "METHOD_TOOLS_LIST",
    "METHOD_TOOLS_CALL",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "outcome_to_mcp",
    "mcp_to_outcome",
]
