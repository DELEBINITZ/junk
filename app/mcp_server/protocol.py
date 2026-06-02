"""Minimal HTTP JSON-RPC MCP protocol handler.

The handler supports the assignment-required lifecycle and tool methods. It is
small on purpose: protocol parsing happens here, while authorization and tool
logic stay in `tools.py`.
"""

from __future__ import annotations

import logging
from typing import Any

from app.db.repository import DataStore
from app.domain import User
from app.mcp_server.tools import ToolError, call_tool, tool_definitions
from app.observability.logging import safe_extra


SERVER_INFO = {"name": "contract-intelligence-mcp", "version": "0.1.0"}
logger = logging.getLogger(__name__)


def handle_mcp_request(payload: dict[str, Any], user: User, store: DataStore) -> dict[str, Any] | None:
    """Handle a single MCP JSON-RPC request for an authenticated user."""

    method = payload.get("method")
    request_id = payload.get("id")
    logger.info(
        "mcp.request.received",
        extra=safe_extra(
            method=method,
            mcp_request_id=request_id,
            user_id=user.id,
            organization_id=user.organization_id,
        ),
    )

    if method == "notifications/initialized":
        logger.debug("mcp.notification.initialized", extra=safe_extra(mcp_request_id=request_id))
        return None

    try:
        if method == "initialize":
            result = {
                "protocolVersion": payload.get("params", {}).get("protocolVersion", "2025-06-18"),
                "serverInfo": SERVER_INFO,
                "capabilities": {"tools": {"listChanged": False}},
            }
        elif method == "tools/list":
            result = {"tools": tool_definitions()}
        elif method == "tools/call":
            params = payload.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ToolError("tools/call requires name and arguments", code=400)
            # User and store come from the server context, not from the tool
            # arguments. This prevents an agent/client from spoofing identity.
            logger.info(
                "mcp.tool.call",
                extra=safe_extra(
                    mcp_request_id=request_id,
                    tool=name,
                    user_id=user.id,
                    organization_id=user.organization_id,
                ),
            )
            result = {"content": [{"type": "json", "json": call_tool(name, arguments, user, store)}]}
        else:
            logger.warning(
                "mcp.method.not_found",
                extra=safe_extra(method=method, mcp_request_id=request_id),
            )
            return _error_response(request_id, -32601, f"Method not found: {method}")
    except ToolError as exc:
        logger.warning(
            "mcp.tool.error",
            extra=safe_extra(method=method, mcp_request_id=request_id, code=exc.code, error=str(exc)),
        )
        return _error_response(request_id, exc.code, str(exc))
    except TypeError as exc:
        logger.warning(
            "mcp.tool.invalid_arguments",
            extra=safe_extra(method=method, mcp_request_id=request_id, error=str(exc)),
        )
        return _error_response(request_id, 400, f"Invalid tool arguments: {exc}")

    logger.info(
        "mcp.request.complete",
        extra=safe_extra(method=method, mcp_request_id=request_id),
    )
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
