"""Minimal MCP (Model Context Protocol) shapes — JSON-RPC 2.0 ``tools/list`` and
``tools/call``. Shared by the in-process client (v1) and the standalone-server /
remote-client transports (promotion path). Keeping the wire shape identical means
promoting a module from in-process to its own ``easm-mcp`` server is a transport
swap, not a contract change.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.contracts import ToolError, ToolOutcome, ToolResult

METHOD_TOOLS_LIST = "tools/list"
METHOD_TOOLS_CALL = "tools/call"


class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Any | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Any | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


def outcome_to_mcp(outcome: ToolOutcome) -> dict[str, Any]:
    """MCP tool-result envelope: text content + structured payload + isError."""
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
    sc = envelope.get("structuredContent") or {}
    if envelope.get("isError"):
        return ToolError(**{**{"code": "remote_error", "message": "remote tool error"}, **sc})
    return ToolResult(**sc) if sc else ToolResult()


def _summarize_result(r: ToolResult) -> str:
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
