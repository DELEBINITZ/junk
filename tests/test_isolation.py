"""Tenant-isolation tests — the hard requirement (0 cross-org leaks)."""

from __future__ import annotations

import pytest

from tests.conftest import make_sc, tool_ctx


@pytest.mark.asyncio
async def test_rag_isolation(services, acme, globex):
    a = await services.deps.rag.retrieve("exposed RDP vpn", collection="reports_kb", ctx=tool_ctx(services, acme))
    assert all(c.org_id == "org_acme" for c in a)
    g = await services.deps.rag.retrieve("confluence CVE", collection="reports_kb", ctx=tool_ctx(services, globex))
    assert all(c.org_id == "org_globex" for c in g)
    assert all(c.doc_id != "R-1001" for c in g)  # acme's confluence report


@pytest.mark.asyncio
async def test_vector_store_requires_org(services):
    with pytest.raises(ValueError):
        await services.deps.rag.store.search("reports_kb", [0.0], org_id="", top_k=5)


@pytest.mark.asyncio
async def test_conversation_isolation(services):
    s = await services.conversations.create_session("org_acme", "u-alice")
    assert await services.conversations.get_session("org_globex", s.id) is None
    assert await services.conversations.get_messages("org_globex", s.id) == []


@pytest.mark.asyncio
async def test_agent_no_cross_org_leak(services, globex):
    r = await services.orchestrator.run_turn(
        globex, question="What critical CVE is on the Confluence server admin.acme.test?"
    )
    assert "CVE-2023-22515" not in r.answer


@pytest.mark.asyncio
async def test_gate_isolation(services):
    ctx = tool_ctx(services, make_sc(roles=("analyst",)))
    await services.mcp.call_tool("trigger_rescan", {"asset": "x"}, ctx)
    assert len(services.action_gate.list_pending("org_acme")) >= 1
    assert services.action_gate.list_pending("org_globex") == []
