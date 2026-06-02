"""MCP boundary: in-process now, standalone server + remote client for promotion."""

from app.core.mcp.inprocess import ActionGateProtocol, InProcessMCPClient, MCPClient
from app.core.mcp.protocol import JSONRPCRequest, JSONRPCResponse, mcp_to_outcome, outcome_to_mcp
from app.core.mcp.remote import RemoteMCPClient
from app.core.mcp.server import make_mcp_app

__all__ = [
    "InProcessMCPClient",
    "MCPClient",
    "ActionGateProtocol",
    "RemoteMCPClient",
    "make_mcp_app",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "outcome_to_mcp",
    "mcp_to_outcome",
]
