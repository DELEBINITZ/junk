"""Demo script to test the full LangGraph agent pipeline.

Usage:
    cd src/
    python3 run_demo.py "What are our current external exposures?"
    python3 run_demo.py "Are any exposed assets mentioned in recent threat reports?"
"""

import asyncio
import sys

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from security_intel.config import Settings
from security_intel.llm.provider import LaneRouter
from security_intel.agents.planner import build_planner
from security_intel.agents.reports.agent import build_reports_agent
from security_intel.agents.easm.agent import build_easm_agent
from security_intel.agents.orchestrator import build_orchestrator


async def main(query: str):
    settings = Settings()

    print(f"LLM endpoint: {settings.llm_base_url}")
    print(f"Models: fast={settings.llm_fast_model} | standard={settings.llm_model} | deep={settings.llm_deep_model}")
    print(f"EASM MCP: {settings.easm_mcp_url or '(using stubs)'}")
    print("-" * 60)
    print(f"Query: {query}")
    print("-" * 60)

    # LaneRouter manages all model tiers
    lane_router = LaneRouter(settings)

    # Build LangGraph agents
    planner = build_planner(lane_router.fast)
    reports_agent = build_reports_agent(lane_router.standard, settings)
    easm_agent = await build_easm_agent(lane_router.standard, settings)

    # Build orchestrator (LangGraph StateGraph — no checkpointer for demo)
    orchestrator = build_orchestrator(
        planner=planner,
        reports_agent=reports_agent,
        easm_agent=easm_agent,
        lane_router=lane_router,
    )

    config = RunnableConfig(
        configurable={
            "thread_id": "demo-session",
            "org_id": "demo-org",
            "user_id": "demo-user",
            "roles": ["admin"],
        }
    )

    input_state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "org_id": "demo-org",
        "user_id": "demo-user",
        "roles": ["admin"],
        "session_id": "demo-session",
        "is_complex": False,
        "plan": None,
        "agent_results": [],
        "final_answer": "",
        "citations": [],
        "blocked": False,
        "block_reason": "",
    }

    print("\nRunning LangGraph orchestrator...")
    result = await orchestrator.ainvoke(input_state, config=config)

    print("\n" + "=" * 60)

    if result.get("blocked"):
        print(f"BLOCKED: {result['block_reason']}")
        return

    print(f"Complexity: {'COMPLEX (deep reasoning)' if result.get('is_complex') else 'SIMPLE (standard)'}")

    plan = result.get("plan")
    if plan:
        print(f"\nPLAN ({len(plan['steps'])} steps):")
        for i, step in enumerate(plan["steps"]):
            deps = f" (after step {step['depends_on']})" if step.get("depends_on") else ""
            print(f"  {i}: [{step['agent']}] {step['task']}{deps}")
        print(f"  Synthesis: {plan['synthesis_goal']}")

    print(f"\nAGENT RESULTS ({len(result.get('agent_results', []))} agents):")
    for r in result.get("agent_results", []):
        print(f"\n  [{r['agent_id']}] ({len(r['tool_calls'])} tool calls)")
        findings = r["findings"]
        if len(findings) > 300:
            findings = findings[:300] + "..."
        print(f"  {findings}")

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(result.get("final_answer", "(no answer)"))


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What are our current external exposures?"
    asyncio.run(main(query))
