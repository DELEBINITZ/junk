"""Remote MCP integration + context-flood guards.

Most of these need NO fastmcp install: they verify the executor HOOK (RBAC + gate
stay local, execution routes remote), the tool-shortlist flood guard, and the new
local easm tool. The last test does a real in-memory FastMCP round trip and SKIPS
if fastmcp isn't installed.
"""

from __future__ import annotations

import pytest

from app.core.contracts import ToolContext, ToolResult
from app.core.mcp.inprocess import InProcessMCPClient
from tests.conftest import make_sc, tool_ctx


class FakeRemote:
    """Stand-in for a FastMCPRemote: records calls, returns a marker result. Lets
    us test the routing hook without standing up a server."""
    transport = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(self, sc):
        return []

    async def call_tool(self, name, arguments, ctx):
        self.calls.append(name)
        return ToolResult(data={"routed": name, "remote": True})


def _mcp_with_remote(services, fake):
    return InProcessMCPClient(
        services.registry, action_gate=services.action_gate,
        logger=services.logger, remote_executors={"easm": fake},
    )


# --------------------------------------------------------------------------- #
# the executor hook — execution routes remote, enforcement stays local
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_remote_executor_routes_read_tool(services, acme):
    fake = FakeRemote()
    mcp = _mcp_with_remote(services, fake)
    out = await mcp.call_tool("get_live_asset_count", {}, tool_ctx(services, acme))
    assert out.ok and out.data.get("remote") is True   # came back from the remote
    assert fake.calls == ["get_live_asset_count"]       # routed there


@pytest.mark.asyncio
async def test_remote_side_effecting_still_gated_locally(services, acme):
    """A remote module's side-effecting tool is STILL stopped by the LOCAL action
    gate — it must never reach the remote server without human approval."""
    fake = FakeRemote()
    mcp = _mcp_with_remote(services, fake)
    out = await mcp.call_tool("trigger_rescan", {"asset": "admin.acme.test"}, tool_ctx(services, acme))
    assert not out.ok and out.code == "requires_approval"
    assert fake.calls == []                              # never dispatched remote


@pytest.mark.asyncio
async def test_remote_rbac_still_enforced_locally(services):
    """RBAC is checked locally BEFORE any remote dispatch — a viewer is denied an
    analyst-only tool and the remote is never touched."""
    viewer = make_sc(roles=("viewer",))
    fake = FakeRemote()
    mcp = _mcp_with_remote(services, fake)
    out = await mcp.call_tool("trigger_rescan", {"asset": "x"}, tool_ctx(services, viewer))
    assert not out.ok and out.code == "forbidden"
    assert fake.calls == []


# --------------------------------------------------------------------------- #
# context-flood guard — specialist shortlists tools
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_specialist_tool_shortlist_caps(services):
    from app.core.agent.specialist import build_specialist

    easm = services.registry.module("easm")
    sp = build_specialist(easm, services.deps, services.mcp)
    selected = await sp._select_tools("how many live assets do we have", cap=2)
    assert len(selected) == 2                            # capped, not all read tools
    # semantic ranking surfaces the asset-count tool for this question
    assert any(t.name == "get_live_asset_count" for t in selected)


# --------------------------------------------------------------------------- #
# the new local easm tool (works on mock today; routes remote when configured)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_get_live_asset_count_local(services, acme):
    out = await services.mcp.call_tool("get_live_asset_count", {}, tool_ctx(services, acme))
    assert out.ok
    assert out.data["live_asset_count"] == 4            # acme mock has 4 exposed assets


# --------------------------------------------------------------------------- #
# real FastMCP in-memory round trip (skips without fastmcp installed)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_fastmcp_inmemory_roundtrip():
    pytest.importorskip("fastmcp")
    from fastmcp import FastMCP

    from app.core.mcp.fastmcp_client import FastMCPRemote

    server = FastMCP("easm-test")

    @server.tool
    async def get_live_asset_count(live_only: bool = True) -> dict:
        return {"live_asset_count": 7, "live_only": live_only}

    remote = FastMCPRemote(server)   # in-memory transport: no network, no auth
    ctx = ToolContext(org_id="org_acme", user_id="u", roles=("viewer",),
                      trace_id="t", request_id="r", deps=None)
    out = await remote.call_tool("get_live_asset_count", {"live_only": True}, ctx)
    assert out.ok
    assert "7" in str(out.data)      # robust to FastMCP's structured-content wrapping
