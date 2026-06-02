"""FastMCP-backed remote transport — the production way to call a promoted module.

WHY FASTMCP: when a capability (EASM, Brand, ACI) becomes its own MCP server, we
don't hand-roll the JSON-RPC plumbing — we use FastMCP, the widely-supported MCP
framework (streamable-HTTP/SSE transports, tool discovery, auth, retries). This
file is a thin ADAPTER: it implements THIS system's ``MCPClient`` interface
(``transport`` / ``list_tools`` / ``call_tool`` — see mcp/inprocess.py) by
delegating to a FastMCP ``Client``. Because the interface matches, the adapter
drops straight into the existing tool boundary as a per-module "remote executor"
(bootstrap wires it; inprocess.py dispatches to it AFTER local RBAC + the action
gate have already run — so promoting a module never weakens those checks).

HOW IDENTITY CROSSES THE WIRE: we do NOT pass org_id as a tool argument. The
trusted local identity (from the ToolContext) is minted into a short-lived,
org-scoped SERVICE TOKEN (security/jwt.create_service_token) and sent as a Bearer
credential; the remote FastMCP server verifies it and re-derives org/roles from
the token. Same isolation rule, end to end.

``fastmcp`` is an OPTIONAL dependency, imported lazily inside ``_client`` — so the
default zero-infra build neither needs nor imports it; it's only loaded when a
module is actually pointed at a remote server.

The ``target`` may be a URL (real server) OR a FastMCP server OBJECT — FastMCP's
in-memory transport lets tests talk to a server with no network, which is how the
test suite exercises this adapter for real.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.core.contracts import ToolContext, ToolError, ToolOutcome, ToolResult
from app.core.security.context import SecurityContext


def _text(res: Any) -> str:
    """Join any text content blocks from a FastMCP CallToolResult — used as a
    human-readable fallback when there's no structured output."""
    parts: list[str] = []
    for block in (getattr(res, "content", None) or []):
        t = getattr(block, "text", None)
        if t:
            parts.append(t)
    return " ".join(parts)


def _to_outcome(res: Any) -> ToolOutcome:
    """Map a FastMCP ``CallToolResult`` back into our ToolResult/ToolError —
    preserving errors-as-data across the network. Prefers the server's structured
    output (so the agent gets typed data); falls back to text content."""
    if getattr(res, "is_error", False):
        return ToolError(code="mcp_error", message=_text(res) or "remote tool error")
    data = getattr(res, "structured_content", None)
    if data is None:
        data = getattr(res, "data", None)
    if isinstance(data, dict):
        # If the server already speaks our shape ({data, citations}), keep it;
        # otherwise treat the whole dict as the tool's structured data.
        if "citations" in data or set(data) <= {"data", "citations", "meta", "ok"}:
            try:
                return ToolResult(**data)
            except Exception:  # noqa: BLE001 - fall through to plain data
                return ToolResult(data=data)
        return ToolResult(data=data)
    if data is not None:
        return ToolResult(data={"result": data})
    return ToolResult(data={"text": _text(res)})


class FastMCPRemote:
    """Adapter implementing the ``MCPClient`` interface over a FastMCP ``Client``.

    One instance per remote module (keyed by module id in the bootstrap executor
    map). ``token_for_ctx`` / ``token_for_sc`` are injected minters that turn the
    trusted local identity into the org-scoped service token sent to the server.
    """

    transport = "fastmcp"

    def __init__(
        self,
        target: Any,
        *,
        token_for_ctx=None,
        token_for_sc=None,
        timeout: float = 30.0,
    ) -> None:
        self._target = target              # a server URL, or a FastMCP server object (tests)
        self._token_for_ctx = token_for_ctx
        self._token_for_sc = token_for_sc
        self._timeout = timeout

    def _client(self, token: str):
        """Build a FastMCP Client for one call. ``fastmcp`` is imported here (lazy)
        so it's only required when a remote module is actually configured. For a
        URL target we attach the org-scoped Bearer token; for an in-memory server
        object (tests) we connect directly."""
        from fastmcp import Client

        if isinstance(self._target, str):
            auth = None
            if token:
                try:
                    from fastmcp.client.auth import BearerAuth

                    auth = BearerAuth(token)
                except Exception:  # noqa: BLE001 - older fastmcp: fall back to no auth helper
                    auth = None
            return Client(self._target, auth=auth, timeout=self._timeout)
        # In-memory transport: hand the Client the server object directly (no network).
        return Client(self._target)

    async def list_tools(self, sc: SecurityContext) -> list[dict]:
        """Ask the remote server which tools it exposes (server applies its own
        org/role filtering using our token). Returned in the same shape as the
        in-process client's ``list_tools`` (name + description + JSON-schema params)."""
        token = self._token_for_sc(sc) if self._token_for_sc else ""
        async with self._client(token) as c:
            tools = await c.list_tools()
        out: list[dict] = []
        for t in tools:
            ann = getattr(t, "annotations", None)
            out.append({
                "name": t.name,
                "description": getattr(t, "description", "") or "",
                "parameters": getattr(t, "inputSchema", None) or {},
                # MCP tool annotations let the server declare a tool's risk; we pass
                # them through so the boundary can decide if a discovered tool is
                # safe to invoke dynamically (read) or must be declared+gated.
                "read_only_hint": getattr(ann, "readOnlyHint", None) if ann else None,
                "destructive_hint": getattr(ann, "destructiveHint", None) if ann else None,
            })
        return out

    async def call_tool(self, name: str, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        """Invoke a tool on the remote server. Transport/connection failures become
        a retriable ToolError (errors-as-data), so a flaky remote never crashes a
        turn. RBAC + the action gate were ALREADY enforced locally before this runs
        (see inprocess.py), and the server re-enforces them too — defense in depth."""
        token = self._token_for_ctx(ctx) if self._token_for_ctx else ""
        # One bounded retry on a transient transport failure (connection reset,
        # brief unavailability) before giving up as errors-as-data. We do NOT retry
        # a tool that ran and returned an error result — only transport faults.
        last_exc: Exception | None = None
        for _attempt in range(2):
            try:
                async with self._client(token) as c:
                    res = await c.call_tool(name, dict(arguments))
                return _to_outcome(res)
            except Exception as exc:  # noqa: BLE001 - network/transport down, server error
                last_exc = exc
        return ToolError(code="mcp_transport_error", message=str(last_exc), retriable=True)

    async def aclose(self) -> None:
        # Clients are created per call and closed by their async context manager,
        # so there is nothing persistent to release here.
        return None


__all__ = ["FastMCPRemote"]
