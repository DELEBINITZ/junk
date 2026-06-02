"""MCP client used by the agent executor."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.db.repository import DataStore
from app.domain import User
from app.mcp_server.protocol import handle_mcp_request


class MCPClientError(Exception):
    pass


class InProcessMCPClient:
    """JSON-RPC client boundary used by the agent.

    The PoC keeps MCP in-process for easy local testing, but the agent still
    speaks JSON-RPC payloads instead of importing tool functions directly. A
    remote HTTP client can replace this class without changing agent logic.
    """

    def __init__(self, user: User, store: DataStore):
        self.user = user
        self.store = store

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool through JSON-RPC and unwrap the JSON content."""

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        response = handle_mcp_request(payload, self.user, self.store)
        if response is None:
            raise MCPClientError("MCP tool call returned no response")
        if "error" in response:
            message = response["error"].get("message", "MCP tool call failed")
            raise MCPClientError(str(message))
        content = response["result"]["content"]
        if not content or content[0].get("type") != "json":
            raise MCPClientError("MCP tool call returned unsupported content")
        return dict(content[0]["json"])
