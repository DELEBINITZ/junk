#!/usr/bin/env python3
"""Routing eval harness — measures whether the orchestrator triggers the RIGHT
agent(s) for a set of golden queries.

This is the missing piece for calling routing "production-grade": you can't tune a
router you don't measure. It runs each golden query through the REAL orchestrator
(same classify → plan → dispatch path production uses) and asserts on which agents
actually ran + whether it stayed agent-free (DIRECT/REFUSE).

Modes:
  (default)   Real LLM. Needs live fast+standard LLM and Qdrant. Measures true
              routing-decision accuracy. No Postgres required.
  --fake-llm  No LLM. Injects a scripted model + stubs Presidio guardrails, then
              runs the real graph (real Qdrant, real tool_call dispatch, real
              reflection). Validates the routing PLUMBING, not decision quality.

Cases whose expected agent isn't registered in this deployment (e.g. EASM without an
MCP server) are SKIPPED, not failed, so the eval adapts to the environment.

Run:  uv run python tests/eval/run_eval.py
      uv run python tests/eval/run_eval.py --json
      uv run python tests/eval/run_eval.py --fake-llm
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GOLDEN = Path(__file__).parent / "golden_queries.json"
FAKE = "--fake-llm" in sys.argv
AS_JSON = "--json" in sys.argv

# In fake mode, the guardrails module must be stubbed BEFORE the orchestrator lazily
# imports it — so do it here, before any security_intel.* graph import.
if FAKE:
    from _fake_llm import install_guardrail_stub

    install_guardrail_stub()


async def _build(cases: list[dict]):
    from security_intel.agents.orchestrator import build_orchestrator
    from security_intel.agents.registry import AgentRegistry
    from security_intel.config import Settings
    from security_intel.llm.provider import Lane, LaneRouter

    settings = Settings()
    lane = LaneRouter(settings)
    registry = AgentRegistry()

    if FAKE:
        from _fake_llm import FakeRoutingLLM, build_decisions, register_light_agents

        fake = FakeRoutingLLM(decisions=build_decisions(cases))
        lane._models = {lane_enum: fake for lane_enum in Lane}
        register_light_agents(registry, settings)
    else:
        from security_intel.main import _register_agents

        await _register_agents(registry, settings, lane_router=lane)

    registry.build_agents(lane.standard)
    orch = build_orchestrator(lane_router=lane, registry=registry)
    return orch, set(registry.agent_ids)


async def _run_case(orch, query: str, idx: int) -> list[str]:
    from langchain_core.runnables import RunnableConfig

    cfg = RunnableConfig(
        configurable={"org_id": "eval", "thread_id": f"eval-{idx}", "user_id": "eval"}
    )
    state = await orch.ainvoke({"user_query": query, "messages": []}, config=cfg)
    return sorted({r["agent_id"] for r in state.get("agent_results", [])})


async def main() -> int:
    cases = json.loads(GOLDEN.read_text())
    orch, available = await _build(cases)

    rows = []
    passed = total = skipped = errored = 0
    for i, case in enumerate(cases):
        exp = sorted(case.get("expect_agents", []))
        missing = [a for a in exp if a not in available]
        if missing:
            skipped += 1
            rows.append({"query": case["query"], "expected": exp, "got": None,
                         "status": "SKIP", "reason": f"agent(s) not registered: {missing}"})
            continue
        try:
            got = await _run_case(orch, case["query"], i)
        except Exception as e:  # noqa: BLE001
            errored += 1
            total += 1
            rows.append({"query": case["query"], "expected": exp, "got": None,
                         "status": "ERROR", "reason": f"{type(e).__name__}: {e}"})
            continue
        ok = got == exp
        total += 1
        passed += int(ok)
        rows.append({"query": case["query"], "expected": exp, "got": got,
                     "status": "PASS" if ok else "FAIL", "class": case.get("class")})

    accuracy = (passed / total * 100) if total else 0.0
    summary = {"mode": "fake-llm" if FAKE else "real-llm", "passed": passed, "total": total,
               "accuracy_pct": round(accuracy, 1), "skipped": skipped, "errored": errored,
               "available_agents": sorted(available)}

    if AS_JSON:
        print(json.dumps({"summary": summary, "rows": rows}, indent=2))
    else:
        mode = "FAKE-LLM (plumbing only)" if FAKE else "REAL-LLM (decision quality)"
        print(f"\nMode: {mode}   Available agents: {sorted(available)}\n")
        print(f"{'STATUS':7} {'EXPECTED':26} {'GOT':26} QUERY")
        print("-" * 100)
        for r in rows:
            exp = ",".join(r["expected"]) or "(none)"
            got = "—" if r["got"] is None else (",".join(r["got"]) or "(none)")
            print(f"{r['status']:7} {exp:26} {got:26} {r['query'][:44]}")
            if r.get("reason"):
                print(f"        └─ {r['reason']}")
        print("-" * 100)
        print(f"Routing accuracy: {passed}/{total} = {accuracy:.0f}%   "
              f"(skipped {skipped}, errored {errored})")

    return 0 if (passed == total and errored == 0) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
