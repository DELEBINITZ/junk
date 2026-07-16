"""Routing PLUMBING regression test — runs in CI with NO external services.

Uses a scripted fake LLM and canned tool_call agents (no Qdrant/TEI) to assert that
the orchestrator correctly EXECUTES a routing decision end-to-end: DIRECT/REFUSE stay
agent-free, SIMPLE dispatches to the named agent. It does NOT test LLM decision
quality (that needs a real model — see tests/eval/run_eval.py against staging).
"""

import sys
from pathlib import Path

import pytest

# Make the package importable without an editable install, and expose _fake_llm
# (which lives next to the eval harness).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent / "eval"))

from _fake_llm import FakeRoutingLLM, install_guardrail_stub  # noqa: E402

# Stub Presidio guardrails BEFORE the orchestrator lazily imports them.
install_guardrail_stub()

from langchain_core.runnables import RunnableConfig  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from security_intel.agents.orchestrator import build_orchestrator  # noqa: E402
from security_intel.agents.registry import AgentRegistry, AgentSpec  # noqa: E402
from security_intel.config import Settings  # noqa: E402
from security_intel.llm.provider import Lane, LaneRouter  # noqa: E402


@tool
async def search_reports(query: str) -> str:
    """canned reports search"""
    return f"[report result for: {query}]"


@tool
async def search_user_guide(query: str) -> str:
    """canned user-guide search"""
    return f"[user-guide result for: {query}]"


def _build_orch(decisions: dict):
    settings = Settings()
    lane = LaneRouter(settings)
    fake = FakeRoutingLLM(decisions=decisions)
    lane._models = {lane_enum: fake for lane_enum in Lane}

    reg = AgentRegistry()
    reg.register(AgentSpec(
        id="reports", display_name="Reports", description="security reports",
        capabilities=["search"], system_prompt="p", tools=[search_reports],
        mode="tool_call", primary_tool="search_reports",
    ))
    reg.register(AgentSpec(
        id="userguide", display_name="User Guide", description="product how-to",
        capabilities=["how-to"], system_prompt="p", tools=[search_user_guide],
        mode="tool_call", primary_tool="search_user_guide",
    ))
    reg.build_agents(lane.standard)
    return build_orchestrator(lane_router=lane, registry=reg)


async def _agents_that_ran(orch, query: str) -> list[str]:
    cfg = RunnableConfig(configurable={"org_id": "t", "thread_id": "t1", "user_id": "u"})
    state = await orch.ainvoke({"user_query": query, "messages": []}, config=cfg)
    return sorted({r["agent_id"] for r in state.get("agent_results", [])})


async def test_direct_uses_no_agents():
    orch = _build_orch({"hi there": {"action": "DIRECT", "response": "hello", "confidence": 0.9}})
    assert await _agents_that_ran(orch, "hi there") == []


async def test_refuse_uses_no_agents():
    orch = _build_orch({"write me code": {"action": "REFUSE", "response": "no", "confidence": 0.9}})
    assert await _agents_that_ran(orch, "write me code") == []


async def test_simple_routes_to_reports():
    orch = _build_orch({"cve details": {"action": "SIMPLE", "agent": "reports",
                                        "task": "cve details", "confidence": 0.9}})
    assert await _agents_that_ran(orch, "cve details") == ["reports"]


async def test_simple_routes_to_userguide():
    orch = _build_orch({"how do I use the dashboard": {"action": "SIMPLE", "agent": "userguide",
                                                       "task": "dashboard", "confidence": 0.9}})
    assert await _agents_that_ran(orch, "how do I use the dashboard") == ["userguide"]


async def test_unknown_agent_falls_through(monkeypatch):
    # Router picks a non-existent agent → must NOT crash; falls through to planner
    # (which, with the fake model producing no plan tool-call, yields a fallback plan).
    orch = _build_orch({"weird": {"action": "SIMPLE", "agent": "does_not_exist",
                                  "task": "weird", "confidence": 0.5}})
    ran = await _agents_that_ran(orch, "weird")
    # Fallback plan routes to the first available agent — the point is: no exception.
    assert isinstance(ran, list)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
