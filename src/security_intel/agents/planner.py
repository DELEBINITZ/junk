from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph


PLANNER_SYSTEM_PROMPT = """You are the Strategic Planner for a security intelligence platform.

Your job: Given a user question, decide which specialist agents to invoke and create an execution plan.

Available agents:
- reports: Searches security reports corpus (threat intel, CVEs, scan findings, remediation guidance). Use for questions about known threats, vulnerabilities, past findings, compliance status.
- easm: Queries external attack surface (assets, exposures, changes, rescans). Use for questions about exposed infrastructure, asset inventory, misconfigurations, surface changes.

Process:
1. First, use describe_reports_agent and/or describe_easm_agent to understand capabilities.
2. Analyze the user question to determine which agents are needed.
3. Call create_execution_plan with your decision.

Principles:
- Use ONE agent when a single domain suffices (most questions).
- Use BOTH only for genuinely cross-domain questions (e.g., "are any of our exposed assets mentioned in recent threat reports?").
- Make each task SPECIFIC and self-contained - the sub-agent only sees its task, not the original question context.
- Set depends_on indices when one step needs another's output (rare - most steps are parallel).
- The synthesis_goal tells the final synthesizer how to combine findings.

ALWAYS end by calling create_execution_plan. Never answer the user question directly."""


@tool
def describe_reports_agent() -> str:
    """Get description of what the Reports Agent can do and when to use it."""
    return (
        "Reports Agent capabilities:\n"
        "- Semantic search over security reports corpus (threat intel, vulnerability assessments, scan results)\n"
        "- Get report metadata (title, date, severity, summary)\n"
        "- Covers: CVEs, threat actor TTPs, remediation steps, compliance findings, scan results\n"
        "- Data source: Organization's ingested security reports (PDFs, threat feeds, scan exports)\n"
        "\n"
        "Use when the question is about: known threats, past findings, vulnerability details, "
        "remediation guidance, compliance status, historical security data."
    )


@tool
def describe_easm_agent() -> str:
    """Get description of what the EASM Agent can do and when to use it."""
    return (
        "EASM Agent capabilities:\n"
        "- Query external assets (domains, IPs, services, ports)\n"
        "- Get current exposures and findings (misconfigurations, expired certs, open panels)\n"
        "- Check attack surface changes over time (new/removed/modified assets)\n"
        "- Trigger rescans of specific assets (side-effecting, requires approval)\n"
        "\n"
        "Use when the question is about: current external infrastructure, exposed services, "
        "asset inventory, misconfigurations, surface changes, what's internet-facing."
    )


@tool
def create_execution_plan(steps: list[dict], synthesis_goal: str) -> str:
    """Create the execution plan for sub-agents.

    Args:
        steps: List of plan steps. Each step is a dict with:
            - agent: 'reports' or 'easm'
            - task: The specific sub-question for that agent (self-contained)
            - depends_on: List of step indices (0-based) that must complete first (usually empty)
        synthesis_goal: How the synthesizer should combine the findings into a final answer.
    """
    plan_summary = f"Plan created with {len(steps)} step(s).\n"
    for i, step in enumerate(steps):
        deps = f" (after step {step.get('depends_on', [])})" if step.get("depends_on") else ""
        plan_summary += f"  Step {i}: [{step['agent']}] {step['task']}{deps}\n"
    plan_summary += f"Synthesis: {synthesis_goal}"
    return plan_summary


def build_planner(llm: ChatOpenAI) -> CompiledStateGraph:
    """Build the Strategic Planner as a ReAct agent with meta-tools."""
    planner_tools = [describe_reports_agent, describe_easm_agent, create_execution_plan]
    return create_react_agent(
        model=llm,
        tools=planner_tools,
        prompt=PLANNER_SYSTEM_PROMPT,
    )
