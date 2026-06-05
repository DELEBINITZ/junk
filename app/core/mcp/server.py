"""Standalone MCP server factory — the OTHER END of the remote transport.

This is the *promotion target* for a module. ``remote.py`` is the client that
calls a promoted module; THIS file builds the server that answers it. You reach
for it when a capability needs independent deploy/scale, separate team
ownership, or a HARD process boundary around adversary-controlled data (e.g. raw
attacker-sourced threat intel you don't want decoded inside the main app).

The pattern: ``app = make_mcp_app(registry_with_one_module, settings, deps)``,
then run that next to the module's datastore (``uvicorn app.core.mcp.server:easm_app``).
The orchestrator points a :class:`RemoteMCPClient` at it and nothing else
changes — because both speak the JSON-RPC protocol from protocol.py.

KEY INSIGHT — the same security still applies. This server does NOT re-implement
RBAC or the action gate. It authenticates the incoming SERVICE TOKEN, rebuilds
the trusted identity FROM that token, and then delegates to an in-process
``InProcessMCPClient`` — which runs the exact same RBAC + action-gate pipeline
described in inprocess.py. Org identity is re-derived from the verified token,
never trusted from arguments, so tenant isolation holds across the boundary.
"""

from __future__ import annotations

# FastAPI at MODULE scope (not inside make_mcp_app): with ``from __future__ import
# annotations`` the route handler's ``request: Request`` annotation is a STRING, and
# FastAPI resolves it against this module's globals — so ``Request`` MUST be importable
# here or FastAPI mis-reads ``request`` as a query parameter and the endpoint breaks.
# FastAPI is a base dependency anyway (the app is built on it), so this costs nothing.
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
from app.core.security.jwt import decode_service_token


def make_mcp_app(
    registry: CapabilityRegistry,
    settings: Settings,
    deps: CoreDeps,
    *,
    title: str = "MCP server",
    audience: str | None = None,
    api_key: str = "",
):
    """Build a FastAPI app exposing the registry's tools over JSON-RPC at /mcp.

    Imports are local so this whole file (and FastAPI) is only loaded when a
    module is actually run as a standalone server. The ``client`` it wraps is an
    ordinary InProcessMCPClient — meaning every tool call this server handles
    still passes through the SAME RBAC + action-gate enforcement.

    TWO auth layers (both enforced before any tool runs):
      * ``api_key`` — the TRANSPORT gate. When set, the request must carry a matching
        ``X-API-Key`` header (constant-time compared) or it's refused. This is the
        "is this my trusted core calling?" check, like a normal production API. Pass
        the same key the caller is configured with (MCP_API_KEY / MCP_API_KEYS).
      * ``audience`` — THIS server's identity (e.g. "easm-mcp"). The bearer SERVICE
        token must carry a matching ``aud`` claim (and be type="service"), or it's
        rejected — so a token minted for another server can't be replayed here. The
        org/user identity is re-derived from that verified token, never from args."""
    import hmac

    app = FastAPI(title=title)
    # Reuse the standard in-process runner: the network layer is just a thin
    # JSON-RPC shell around the identical enforcement path.
    client = InProcessMCPClient(registry, action_gate=deps.action_gate, logger=deps.logger)

    def _sc(request: Request) -> SecurityContext:
        """Authenticate the request and rebuild the TRUSTED identity from its
        token. This is the linchpin of cross-service tenant isolation: we read
        the bearer token, cryptographically verify+decode it (``decode_token``),
        and construct the SecurityContext purely from its CLAIMS — org, user,
        roles. Nothing here ever reads identity from the request body, so a
        caller cannot assert a different org than its token grants."""
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        # SERVICE token only: verified with the service key (the PUBLIC key when an
        # asymmetric keypair is configured, so this server can't mint), must be
        # type="service", and bound to THIS server's audience.
        claims = decode_service_token(settings, token, expected_audience=audience)
        return SecurityContext(
            org_id=str(claims["org_id"]), user_id=str(claims["sub"]),
            roles=tuple(claims.get("roles", []) or ("viewer",)),
            email=str(claims.get("email", "")), token_id=str(claims.get("jti", "")),
        )

    @app.post("/mcp")
    async def mcp(request: Request):
        """The single JSON-RPC endpoint. It authenticates first, then dispatches
        on ``method`` to the matching tool-boundary operation, and always replies
        with a JSON-RPC envelope (using standard-ish negative error codes)."""
        body = await request.json()
        rid = body.get("id")                      # echo the caller's id back for correlation
        # LAYER 1 — TRANSPORT: verify the per-server API key (constant-time) before
        # touching the body's identity or any tool. Generic error: never reveal which
        # check failed.
        if api_key and not hmac.compare_digest(request.headers.get("x-api-key", ""), api_key):
            return JSONResponse(JSONRPCResponse(
                id=rid, error={"code": -32001, "message": "auth failed"}).model_dump())
        # LAYER 2 — IDENTITY: verify the service token and rebuild sc. The identity
        # built here is the ONLY source of org/roles below.
        try:
            sc = _sc(request)
        except Exception:  # noqa: BLE001
            return JSONResponse(JSONRPCResponse(
                id=rid, error={"code": -32001, "message": "auth failed"}).model_dump())

        method = body.get("method")
        params = body.get("params", {}) or {}
        if method == METHOD_TOOLS_LIST:
            # List the tools visible to this verified caller.
            return JSONResponse(JSONRPCResponse(
                id=rid, result={"tools": await client.list_tools(sc)}).model_dump())
        if method == METHOD_TOOLS_CALL:
            # Build the trusted ToolContext STRICTLY from the token-derived sc —
            # org/user/roles come from sc, only the tool name + args come from the
            # request body. Then run it through the in-process client, which is
            # where RBAC and the action gate are enforced (see inprocess.py).
            ctx = ToolContext(
                org_id=sc.org_id, user_id=sc.user_id, roles=sc.roles,
                trace_id=body.get("id", "mcp"), request_id=str(rid), deps=deps,
            )
            outcome = await client.call_tool(params.get("name", ""), params.get("arguments", {}), ctx)
            # Pack the ToolResult/ToolError back into the MCP result envelope.
            return JSONResponse(JSONRPCResponse(id=rid, result=outcome_to_mcp(outcome)).model_dump())
        # Unknown method -> JSON-RPC "method not found" (-32601).
        return JSONResponse(JSONRPCResponse(
            id=rid, error={"code": -32601, "message": f"method not found: {method}"}).model_dump())

    @app.get("/healthz")
    async def healthz():
        """Liveness probe for orchestration (k8s/load balancer): reports which
        modules this server is hosting."""
        return {"status": "ok", "modules": [m.id for m in registry.modules()]}

    return app


__all__ = ["make_mcp_app"]
