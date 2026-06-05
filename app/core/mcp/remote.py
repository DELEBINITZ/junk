"""Remote MCP client — the "remote transport" half of the tool boundary.

When a module is promoted to its own service (``easm-mcp``, ``brand-mcp``,
``aci-mcp``), the agent still needs to call its tools. This client does that over
HTTP using the JSON-RPC envelopes from protocol.py, while exposing the EXACT same
``list_tools`` / ``call_tool`` methods as ``InProcessMCPClient``. So the
orchestrator swaps in-process for remote with no other code change — the only
difference is that the call now crosses a process/network boundary instead of
staying inline.

SECURITY — how trust crosses the wire. We do NOT pass the org id as data. The
trusted identity (from the local ToolContext) is minted into a short-lived,
org-scoped SERVICE TOKEN, sent in the ``Authorization`` header, and the remote
server re-derives ``org_id`` from THAT token (see server.py). So the same rule
as in-process holds end to end: org/identity come from a verified token, never
from tool arguments — a remote server can't be tricked into crossing tenants.

Note: the RBAC + action-gate enforcement runs on the SERVER side (its own
InProcessMCPClient), exactly as it would locally. This client is just the
transport; it never weakens those checks.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import Any

from app.core.contracts import ToolContext, ToolError, ToolOutcome
from app.core.mcp.protocol import (
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
    JSONRPCResponse,
    mcp_to_outcome,
)
from app.core.security.context import SecurityContext

# Two token minters, injected at construction. Each takes a trusted local
# identity (a ToolContext for tool calls, a SecurityContext for listing) and
# returns an org-scoped service token to authenticate the remote request. They
# are passed in rather than hardcoded so token policy lives in one place.
TokenProvider = Callable[[ToolContext], str]
SCTokenProvider = Callable[[SecurityContext], str]


class RemoteMCPClient:
    """Speaks MCP-over-HTTP to one promoted module's server. Same interface as
    the in-process client; ``transport = "remote"`` distinguishes it in traces."""

    transport = "remote"

    def __init__(
        self,
        base_url: str,
        token_for_ctx: TokenProvider,
        token_for_sc: SCTokenProvider,
        *,
        timeout: float = 30.0,
        api_key: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")   # the remote MCP server's address
        self._token_for_ctx = token_for_ctx     # mint a service token for tool CALLS
        self._token_for_sc = token_for_sc       # mint a service token for tool LISTING
        self._timeout = timeout
        self._api_key = api_key                 # transport API key (X-API-Key); "" => none
        self._client = None                     # lazy httpx.AsyncClient (see _http)

    def _http(self):
        """Lazily create and reuse one HTTP client (connection pooling, and httpx
        is only imported when a remote module is actually configured)."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _rpc(self, method: str, params: dict, token: str) -> JSONRPCResponse:
        """One JSON-RPC round trip to the remote ``/mcp`` endpoint. Builds the
        request envelope (a fresh random ``id`` correlates the reply), attaches
        the org-scoped service token as a bearer credential, and parses the reply
        back into a typed JSONRPCResponse."""
        payload = {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}
        headers = {"Authorization": f"Bearer {token}"}      # identity travels here, not in params
        if self._api_key:
            headers["X-API-Key"] = self._api_key            # transport gate (server checks it)
        r = await self._http().post(f"{self._base_url}/mcp", json=payload, headers=headers)
        r.raise_for_status()
        return JSONRPCResponse(**r.json())

    async def list_tools(self, sc: SecurityContext) -> list[dict]:
        """Remote ``tools/list``: ask the server which tools this caller may see.
        The server applies its own org/role filtering using the token we mint."""
        resp = await self._rpc(METHOD_TOOLS_LIST, {}, self._token_for_sc(sc))
        return (resp.result or {}).get("tools", [])

    async def call_tool(self, name: str, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        """Remote ``tools/call``: invoke a named tool on the remote server and
        translate its reply back into a ToolResult/ToolError. Note how failures
        become DATA, preserving the errors-as-data contract across the network:
          * transport/HTTP failure -> a retriable ``mcp_transport_error``;
          * a JSON-RPC ``error`` reply -> an ``mcp_error``;
          * otherwise decode the result envelope via ``mcp_to_outcome``.
        RBAC and the action gate were already enforced ON THE SERVER, so a remote
        side-effecting call comes back as ``requires_approval`` just like locally.
        """
        try:
            resp = await self._rpc(
                METHOD_TOOLS_CALL, {"name": name, "arguments": dict(arguments)},
                self._token_for_ctx(ctx),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolError(code="mcp_transport_error", message=str(exc), retriable=True)
        if resp.error:
            return ToolError(code="mcp_error", message=str(resp.error.get("message", "")),
                             details=resp.error)
        return mcp_to_outcome(resp.result or {})

    async def aclose(self) -> None:
        """Close and drop the pooled HTTP client; safe if it was never created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["RemoteMCPClient", "TokenProvider", "SCTokenProvider"]
