"""Hierarchical specialist sub-agents: tool isolation + parallel cross-pillar."""

from __future__ import annotations

import pytest

from app.core.agent.specialist import GenericSpecialist, build_specialist


def test_specialists_are_scoped_to_their_own_tools(services):
    """The scaling guarantee: each specialist holds ONLY its module's tools, so
    tool schemas never co-locate in one agent's context."""
    easm = services.registry.module("easm")
    aci = services.registry.module("aci")
    sp_e = build_specialist(easm, services.deps, services.mcp)
    sp_a = build_specialist(aci, services.deps, services.mcp)
    e_tools = {t.name for t in sp_e._read_tools()}
    a_tools = {t.name for t in sp_a._read_tools()}
    assert "query_assets" in e_tools and "get_threat_actors" in a_tools
    assert e_tools.isdisjoint(a_tools)  # no overlap — tools never co-locate
    # supervisor never sees the union; each specialist sees only its slice
    assert len(e_tools) <= len(easm.tools) and len(a_tools) <= len(aci.tools)


@pytest.mark.asyncio
async def test_specialist_investigate_returns_own_pillar_only(services, acme):
    from tests.conftest import tool_ctx

    easm = services.registry.module("easm")
    sp = build_specialist(easm, services.deps, services.mcp)
    res = await sp.investigate("what is exposed on our attack surface?", tool_ctx(services, acme))
    assert res.module_id == "easm"
    assert res.chunks and all(c.source == "easm" for c in res.chunks)


@pytest.mark.asyncio
async def test_parallel_cross_pillar_join(services, acme):
    r = await services.orchestrator.run_turn(
        acme, question="what is our biggest exposure and which threat actor weaponizes it?"
    )
    assert len(r.route_modules) >= 2  # supervisor fanned out
    sources = {c.get("source") for c in r.citations}
    assert len(sources) >= 2  # answer cited across pillars (e.g. easm + aci)


@pytest.mark.asyncio
async def test_single_module_behavior_preserved(services, acme):
    """1 routed module => exactly one specialist => same as before the refactor."""
    r = await services.orchestrator.run_turn(acme, question="what critical CVE is on our confluence server?")
    assert "CVE-2023-22515" in r.answer and "reports" in r.route_modules


def test_generic_specialist_is_default(services):
    reports = services.registry.module("reports")
    sp = build_specialist(reports, services.deps, services.mcp)
    assert isinstance(sp, GenericSpecialist)  # no custom specialist declared -> generic
