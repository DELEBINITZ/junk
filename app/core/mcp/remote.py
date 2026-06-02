"""Remote MCP client — calls a module promoted to its own MCP server
(``easm-mcp``, ``brand-mcp``, ``aci-mcp``) over HTTP JSON-RPC.

Identical interface to :class:`InProcessMCPClient`; the orchestrator swaps one
for the other with no change. The trusted org identity is propagated as a
short-lived, org-scoped **service token** (never as a tool arg); the remote
server re-derives ``org_id`` from it.
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

# Mints an org-scoped service token from the trusted context.
TokenProvider = Callable[[ToolContext], str]
SCTokenProvider = Callable[[SecurityContext], str]


class RemoteMCPClient:
    transport = "remote"

    def __init__(
        self,
        base_url: str,
        token_for_ctx: TokenProvider,
        token_for_sc: SCTokenProvider,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token_for_ctx = token_for_ctx
        self._token_for_sc = token_for_sc
        self._timeout = timeout
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _rpc(self, method: str, params: dict, token: str) -> JSONRPCResponse:
        payload = {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}
        r = await self._http().post(
            f"{self._base_url}/mcp", json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return JSONRPCResponse(**r.json())

    async def list_tools(self, sc: SecurityContext) -> list[dict]:
        resp = await self._rpc(METHOD_TOOLS_LIST, {}, self._token_for_sc(sc))
        return (resp.result or {}).get("tools", [])

    async def call_tool(self, name: str, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
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
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["RemoteMCPClient", "TokenProvider", "SCTokenProvider"]
