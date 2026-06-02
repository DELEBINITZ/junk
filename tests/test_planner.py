"""Planner-mode orchestration: retrieval-as-tool, LLM-brain planning (deterministic
fallback), dependency-aware dispatch, and cross-module synthesis.

These run on the zero-infra deterministic path, so the LLM planner falls back to
the heuristic plan (one parallel step per routed module). That still exercises the
full planner graph: plan -> plan_dispatch -> replan_gate -> answer.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from app.config import reload_settings
from app.core.agent.planner import Planner
from app.core.bootstrap import build_services, seed_demo
from tests.conftest import tool_ctx


# --------------------------------------------------------------------------- #
# a services bundle wired for orchestrator_mode=planner
# --------------------------------------------------------------------------- #
@pytest_asyncio.fixture
async def planner_services():
    os.environ["ORCHESTRATOR_MODE"] = "planner"
    svc = build_services(reload_settings())
    await seed_demo(svc)
    try:
        yield svc
    finally:
        await svc.aclose()
        os.environ.pop("ORCHESTRATOR_MODE", None)
        reload_settings()


# --------------------------------------------------------------------------- #
# P1 — retrieval-as-a-tool
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_search_reports_is_a_tool(services, acme):
    """The reports RAG search is now callable as a tool through the MCP boundary,
    org-scoped from the trusted context."""
    out = await services.mcp.call_tool("search_reports", {"query": "confluence CVE"}, tool_ctx(services, acme))
    assert out.ok
    assert out.data.get("hits", 0) >= 1
    assert out.citations  # grounded passages came back


def test_search_reports_not_auto_invoked(services):
    """auto_invoke=False keeps the heuristic gatherer from firing search_reports
    (the bound retriever already auto-gathers; the tool is for the LLM/planner)."""
    reports = services.registry.module("reports")
    assert reports.tools["search_reports"].auto_invoke is False
    assert reports.tools["find_expiring_items"].auto_invoke is True


# --------------------------------------------------------------------------- #
# P2 — the planner produces a plan
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_heuristic_plan_one_step_per_domain(services, acme):
    planner = Planner(services.registry, services.deps.llm, services.deps.settings)
    plan = await planner.plan("what critical CVE is on our confluence server?", acme)
    assert plan.steps
    assert "reports" in [s.domain for s in plan.steps]
    # deterministic plan has no cross-step dependencies
    assert all(s.depends_on == [] for s in plan.steps)


def test_plan_validation_drops_bad_domains_and_breaks_cycles(services):
    """The validator keeps only known domains and only depends_on ids that refer
    to EARLIER steps — so a malformed/cyclic LLM plan becomes safe + acyclic."""
    planner = Planner(services.registry, services.deps.llm, services.deps.settings)
    raw = {
        "steps": [
            {"id": "s1", "domain": "reports", "subq": "a", "depends_on": ["s2"]},  # forward dep -> dropped
            {"id": "s2", "domain": "nope", "subq": "b"},                            # unknown domain -> dropped
            {"id": "s3", "domain": "easm", "subq": "c", "depends_on": ["s1"]},      # back dep -> kept
        ],
        "synthesis": "x",
    }
    plan = planner._validate(raw, available=["reports", "easm", "aci"])
    ids = {s.id: s for s in plan.steps}
    assert "s2" not in ids                      # unknown domain dropped
    assert ids["s1"].depends_on == []           # forward dependency stripped
    assert ids["s3"].depends_on == ["s1"]       # valid backward dependency kept


# --------------------------------------------------------------------------- #
# P3 — planner-mode turns end to end (built-in engine, planner graph)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_planner_mode_single_module_grounded(planner_services, acme):
    """A reports question, answered through the FULL planner graph, still grounds
    and cites — same outcome as heuristic mode, different orchestration."""
    r = await planner_services.orchestrator.run_turn(
        acme, question="what critical CVE is on our confluence server?"
    )
    assert "CVE-2023-22515" in r.answer
    assert "reports" in r.route_modules


@pytest.mark.asyncio
async def test_planner_mode_cross_pillar(planner_services, acme):
    """A cross-domain question fans the plan across >= 2 modules and the answer
    cites across pillars."""
    r = await planner_services.orchestrator.run_turn(
        acme, question="what is our biggest exposure and which threat actor weaponizes it?"
    )
    assert len(r.route_modules) >= 2
    sources = {c.get("source") for c in r.citations}
    assert len(sources) >= 2


@pytest.mark.asyncio
async def test_planner_mode_isolation(planner_services, globex):
    """Tenant isolation holds in planner mode: globex never sees acme's confluence
    report (org_id is enforced inside every step's retrieval)."""
    r = await planner_services.orchestrator.run_turn(
        globex, question="what critical CVE is on our confluence server?"
    )
    assert "CVE-2023-22515" not in r.answer


@pytest.mark.asyncio
async def test_dependency_dispatch_executes_both_waves(planner_services, acme):
    """The headline P3 capability: a hand-built plan where step s2 depends_on s1
    executes in two waves (s1 then s2), and BOTH steps contribute findings — i.e.
    the cross-module dependency chain actually runs, upstream-then-downstream."""
    from app.core.agent.nodes import plan_dispatch_node
    from app.core.agent.state import AgentContext

    o = planner_services.orchestrator
    ctx = AgentContext(
        deps=o.deps, sc=acme, tool_ctx=tool_ctx(planner_services, acme), mcp=o.mcp,
        registry=o.registry, input_guard=o.input_guard, output_guard=o.output_guard,
        settings=o.settings, supervisor=o.supervisor,
    )
    state = {
        "safe_question": "biggest exposure and who weaponizes it",
        "plan": [
            {"id": "s1", "domain": "easm", "subq": "our biggest external exposure", "depends_on": []},
            {"id": "s2", "domain": "aci", "subq": "which actor weaponizes it", "depends_on": ["s1"]},
        ],
    }
    out = await plan_dispatch_node(state, ctx)
    by_id = {r["id"]: r for r in out["plan_results"]}
    assert by_id["s1"]["ok"] and by_id["s2"]["ok"]   # both waves executed
    assert out["context_chunks"]                     # merged evidence from both pillars
