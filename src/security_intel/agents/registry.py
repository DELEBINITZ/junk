"""Agent Registry — single-point registration for all specialist agents.

Adding a new agent requires:
1. Add MCP server config to MCP_SERVERS env var (or add local tools)
2. Register the agent with description and capabilities here
3. That's it — planner auto-discovers available agents, orchestrator auto-routes.

No changes to planner prompts or orchestrator routing needed.
"""

from dataclasses import dataclass, field
from typing import Callable, Awaitable

from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph

from security_intel.config import Settings
from security_intel.observability.logging import get_logger

logger = get_logger("registry")


@dataclass
class AgentSpec:
    """Specification for a specialist agent."""

    id: str
    display_name: str
    description: str
    capabilities: list[str]
    system_prompt: str
    tools: list[BaseTool] = field(default_factory=list)
    side_effecting_tools: set[str] = field(default_factory=set)
    min_role: str = "viewer"


class AgentRegistry:
    """Registry of all specialist agents. Agents register at startup, planner queries at runtime."""

    def __init__(self):
        self._specs: dict[str, AgentSpec] = {}
        self._agents: dict[str, CompiledStateGraph] = {}

    def register(self, spec: AgentSpec) -> None:
        """Register an agent specification."""
        self._specs[spec.id] = spec
        logger.info(f"Registered agent: {spec.id} ({spec.display_name})")

    def build_agents(self, llm: ChatOpenAI) -> dict[str, CompiledStateGraph]:
        """Build all registered agents as LangGraph ReAct agents."""
        for agent_id, spec in self._specs.items():
            if not spec.tools:
                logger.warning(f"Agent '{agent_id}' has no tools, skipping build")
                continue

            agent = create_react_agent(
                model=llm,
                tools=spec.tools,
                prompt=spec.system_prompt,
            )
            self._agents[agent_id] = agent
            logger.info(f"Built agent: {agent_id} ({len(spec.tools)} tools)")

        return self._agents

    def get_agent(self, agent_id: str) -> CompiledStateGraph | None:
        """Get a built agent by ID."""
        return self._agents.get(agent_id)

    def get_spec(self, agent_id: str) -> AgentSpec | None:
        """Get agent specification by ID."""
        return self._specs.get(agent_id)

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agents.keys())

    @property
    def specs(self) -> dict[str, AgentSpec]:
        return self._specs

    def build_planner_tools(self) -> list[BaseTool]:
        """Auto-generate planner meta-tools from registered agents.

        Each agent gets a describe_<id>_agent tool + one create_execution_plan tool.
        Planner uses these to decide which agents to invoke.
        """
        planner_tools = []

        for agent_id, spec in self._specs.items():
            if agent_id not in self._agents:
                continue

            # Create a describe tool for this agent
            desc_tool = _make_describe_tool(agent_id, spec)
            planner_tools.append(desc_tool)

        # The plan creation tool
        planner_tools.append(_make_plan_tool(list(self._agents.keys())))
        return planner_tools

    def build_planner_system_prompt(self) -> str:
        """Auto-generate planner system prompt from registered agents."""
        agent_descriptions = []
        for agent_id, spec in self._specs.items():
            if agent_id not in self._agents:
                continue
            capabilities = ", ".join(spec.capabilities[:3])
            agent_descriptions.append(
                f"- {agent_id}: {spec.description} Capabilities: {capabilities}"
            )

        agents_block = "\n".join(agent_descriptions)

        return f"""You are the Strategic Planner for a security intelligence platform.

Your job: Understand the user's INTENT, then create a precise execution plan for specialist agents.
Note: Greetings, chitchat, and non-security queries are already handled before reaching you — you only receive queries that genuinely need security data.

## Available Agents
{agents_block}

## How to Think
1. **Parse intent**: What does the user actually need? (information? action? comparison?)
2. **Scope check**: Which domain(s) does this touch? (threat intel? attack surface? both?)
3. **Decompose**: Break into specific, self-contained sub-questions for each agent.
4. **Optimize**: Can agents run in parallel? Does one need another's output first?

## Task Writing Rules
- Each task must be SELF-CONTAINED — the sub-agent sees ONLY its task string
- Include specific entities (CVE IDs, hostnames, terms) the user mentioned
- Frame as a clear question or directive, not a vague exploration
- Remind agents to present findings in a friendly, clear manner
- BAD: "Look into threats" → GOOD: "Search for reports about CVE-2024-1234 including severity, affected systems, and remediation steps"
- BAD: "Check surface" → GOOD: "List all exposed assets with CRITICAL or HIGH severity findings, including hostnames and specific vulnerabilities"

## Decision Rules
- ONE agent: single domain, straightforward question (80% of queries)
- MULTIPLE parallel: genuinely cross-domain ("are exposed assets in threat reports?")
- SEQUENTIAL (depends_on): output of one agent needed by another (rare — <5% of queries)

## Conversation Awareness
- Follow-up questions ("what about X?", "tell me more") → infer context from prior messages
- If user references prior findings, include that context in the task

ALWAYS call create_execution_plan. Never answer the user's question directly."""


def _make_describe_tool(agent_id: str, spec: AgentSpec) -> BaseTool:
    """Generate a describe tool for the planner."""
    capabilities_text = "\n".join(f"  - {c}" for c in spec.capabilities)
    description_text = (
        f"{spec.display_name} capabilities:\n{capabilities_text}\n\nUse when: {spec.description}"
    )

    @tool
    def describe() -> str:
        """Get description of what this agent can do and when to use it."""
        return description_text

    describe.name = f"describe_{agent_id}_agent"
    describe.description = (
        f"Get description of what the {spec.display_name} can do and when to use it."
    )
    return describe


def _make_plan_tool(available_agents: list[str]) -> BaseTool:
    """Generate the create_execution_plan tool with available agent IDs."""
    agents_str = ", ".join(f"'{a}'" for a in available_agents)

    @tool
    def create_execution_plan(steps: list[dict], synthesis_goal: str) -> str:
        f"""Create the execution plan for sub-agents.

        Args:
            steps: List of plan steps. Each step is a dict with:
                - agent: One of [{agents_str}]
                - task: The specific sub-question for that agent (self-contained)
                - depends_on: List of step indices (0-based) that must complete first
            synthesis_goal: How the synthesizer should combine the findings.
        """
        plan_summary = f"Plan created with {len(steps)} step(s).\n"
        for i, step in enumerate(steps):
            deps = f" (after step {step.get('depends_on', [])})" if step.get("depends_on") else ""
            plan_summary += f"  Step {i}: [{step['agent']}] {step['task']}{deps}\n"
        plan_summary += f"Synthesis: {synthesis_goal}"
        return plan_summary

    create_execution_plan.__doc__ = (
        f"Create execution plan. Available agents: [{agents_str}]. "
        "Each step has: agent, task, depends_on (list of prior step indices)."
    )
    return create_execution_plan
