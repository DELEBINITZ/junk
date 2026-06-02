"""Standalone MCP server factory — the *promotion target* for a module.

When a capability needs its own deploy/scale, team ownership, or a hard security
boundary around adversary-controlled data, package its tools as a standalone MCP
server: ``app = make_mcp_app(registry_with_one_module, settings, deps)`` and run
it next to that module's datastore. The orchestrator points a
:class:`RemoteMCPClient` at it; nothing else changes.

Org identity is re-derived from the verified service token — never trusted from
arguments. Run e.g.::  uvicorn app.core.mcp.server:easm_app
"""

from __future__ import annotations

from app.config import Settings
from app.core.contracts import CoreDeps, ToolContext
from app.core.mcp.inprocess import InProcessMCPClient
from app.core.mcp.protocol import (
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
    JSONRPCResponse,
    outcome_to_mcp,
)
from app.core.registry import CapabilityRegistry
from app.core.security.context import SecurityContext
from app.core.security.jwt import decode_token


def make_mcp_app(
    registry: CapabilityRegistry,
    settings: Settings,
    deps: CoreDeps,
    *,
    title: str = "MCP server",
):
    """Build a FastAPI app exposing the registry's tools over JSON-RPC at /mcp."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title=title)
    client = InProcessMCPClient(registry, action_gate=deps.action_gate, logger=deps.logger)

    def _sc(request: Request) -> SecurityContext:
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        claims = decode_token(settings, token, expected_type="access")
        return SecurityContext(
            org_id=str(claims["org_id"]), user_id=str(claims["sub"]),
            roles=tuple(claims.get("roles", []) or ("viewer",)),
            email=str(claims.get("email", "")), token_id=str(claims.get("jti", "")),
        )

    @app.post("/mcp")
    async def mcp(request: Request):
        body = await request.json()
        rid = body.get("id")
        try:
            sc = _sc(request)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(JSONRPCResponse(
                id=rid, error={"code": -32001, "message": f"auth failed: {exc}"}).model_dump())

        method = body.get("method")
        params = body.get("params", {}) or {}
        if method == METHOD_TOOLS_LIST:
            return JSONResponse(JSONRPCResponse(
                id=rid, result={"tools": await client.list_tools(sc)}).model_dump())
        if method == METHOD_TOOLS_CALL:
            ctx = ToolContext(
                org_id=sc.org_id, user_id=sc.user_id, roles=sc.roles,
                trace_id=body.get("id", "mcp"), request_id=str(rid), deps=deps,
            )
            outcome = await client.call_tool(params.get("name", ""), params.get("arguments", {}), ctx)
            return JSONResponse(JSONRPCResponse(id=rid, result=outcome_to_mcp(outcome)).model_dump())
        return JSONResponse(JSONRPCResponse(
            id=rid, error={"code": -32601, "message": f"method not found: {method}"}).model_dump())

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "modules": [m.id for m in registry.modules()]}

    return app


__all__ = ["make_mcp_app"]
