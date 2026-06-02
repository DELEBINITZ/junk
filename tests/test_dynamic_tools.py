"""Pure-dynamic MCP tools: the agent discovers an EXTERNAL server's toolset at
runtime (no local declaration), shortlists it, and can call the read-safe ones —
while destructive tools are excluded from the dynamic path (must be declared+gated).
"""

from __future__ import annotations

import pytest

from app.core.contracts import ToolResult
from app.core.mcp.inprocess import InProcessMCPClient
from tests.conftest import make_sc, tool_ctx


class DiscoverRemote:
    """Fake external MCP server (the FastMCPRemote stand-in) that ADVERTISES tools
    via list_tools — including read-only and destructive ones — and records calls."""
    transport = "fake"

    def __init__(self) -> None:
        self.called: list[str] = []

    async def list_tools(self, sc):
        return [
            {"name": "get_scan_status", "description": "Status of the latest external scan.",
             "parameters": {"type": "object", "properties": {}},
             "read_only_hint": True, "destructive_hint": None},
            {"name": "purge_assets", "description": "Permanently delete assets.",
             "parameters": {}, "read_only_hint": False, "destructive_hint": True},
        ]

    async def call_tool(self, name, arguments, ctx):
        self.called.append(name)
        return ToolResult(data={"tool": name, "ran": True})


def _mcp(services, ex):
    return InProcessMCPClient(
        services.registry, action_gate=services.action_gate, logger=services.logger,
        remote_executors={"easm": ex}, settings=services.settings,
    )


@pytest.mark.asyncio
async def test_discovery_excludes_destructive(services, acme):
    mcp = _mcp(services, DiscoverRemote())
    found = {d["name"] for d in await mcp.discover_tools("easm", tool_ctx(services, acme))}
    assert "get_scan_status" in found          # read-only -> discoverable + callable
    assert "purge_assets" not in found         # destructive -> excluded from dynamic path


@pytest.mark.asyncio
async def test_call_discovered_read_tool_routes_to_server(services, acme):
    ex = DiscoverRemote()
    mcp = _mcp(services, ex)
    await mcp.discover_tools("easm", tool_ctx(services, acme))     # populate the resolve map
    out = await mcp.call_tool("get_scan_status", {}, tool_ctx(services, acme))
    assert out.ok and out.data["tool"] == "get_scan_status"
    assert ex.called == ["get_scan_status"]                        # executed on the server


@pytest.mark.asyncio
async def test_destructive_remote_tool_not_callable_dynamically(services, acme):
    ex = DiscoverRemote()
    mcp = _mcp(services, ex)
    await mcp.discover_tools("easm", tool_ctx(services, acme))
    out = await mcp.call_tool("purge_assets", {}, tool_ctx(services, acme))
    assert not out.ok and out.code == "unknown_tool"               # never registered as safe
    assert ex.called == []                                         # and never reached the server


@pytest.mark.asyncio
async def test_dynamic_rbac_floor(services):
    """A dynamically-discovered tool still passes a local role floor before the
    remote call (the server then enforces fine-grained RBAC)."""
    ex = DiscoverRemote()
    mcp = _mcp(services, ex)
    # default floor is "viewer"; an empty-roles caller doesn't meet it
    noroles = make_sc(roles=())
    await mcp.discover_tools("easm", tool_ctx(services, noroles))
    out = await mcp.call_tool("get_scan_status", {}, tool_ctx(services, noroles))
    assert not out.ok and out.code == "forbidden"
    assert ex.called == []


@pytest.mark.asyncio
async def test_specialist_unifies_local_and_discovered(services, acme):
    """The specialist's candidate set merges local read tools with discovered
    remote ones (this is what the tool-calling LLM chooses from)."""
    from app.core.agent.specialist import build_specialist

    mcp = _mcp(services, DiscoverRemote())
    easm = services.registry.module("easm")
    sp = build_specialist(easm, services.deps, mcp)
    cands = {c["name"] for c in await sp._gather_candidates("status of our latest scan", tool_ctx(services, acme), cap=10)}
    assert "get_scan_status" in cands              # discovered tool is a candidate
    assert "query_assets" in cands                 # local tool still present
    assert "purge_assets" not in cands             # destructive never surfaced
