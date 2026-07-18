"""Agent Registry — single-point registration for all specialist agents.

Adding a new agent requires:
1. Add MCP server config to MCP_SERVERS env var (or add local tools)
2. Register the agent with description and capabilities here
3. That's it — planner auto-discovers available agents, orchestrator auto-routes.

No changes to planner prompts or orchestrator routing needed.
"""

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from security_intel.config import Settings
from security_intel.observability.logging import get_logger
from security_intel.prompts.planner import PLANNER_SYSTEM_TEMPLATE

logger = get_logger("registry")


@dataclass
class AgentSpec:
    """Specification for a specialist agent.

    ``mode`` controls execution:
    - "react": full create_react_agent tool-reasoning loop. Use when the agent
      genuinely chains/chooses tools or does multi-step work (e.g. EASM).
    - "tool_call": a single deterministic call to ``primary_tool`` (no LLM loop).
      Use for pure-retrieval agents whose job is "run one search and return" —
      cuts an entire LLM round-trip and its failure modes per query.
    """

    id: str
    display_name: str
    description: str
    capabilities: list[str]
    # Optional — react agents auto-registered from config (e.g. MCP servers) leave this
    # blank and get a generated prompt (see prompts/agent.render_agent_system_prompt).
    system_prompt: str = ""
    tools: list[BaseTool] = field(default_factory=list)
    side_effecting_tools: set[str] = field(default_factory=set)
    min_role: str = "viewer"
    mode: str = "react"  # "react" | "tool_call"
    primary_tool: str = ""  # required for mode="tool_call": the tool to invoke
    primary_tool_arg: str = "query"  # the kwarg the task string is passed as


class AgentRegistry:
    """Registry of all specialist agents. Agents register at startup, planner queries at runtime."""

    def __init__(self):
        self._specs: dict[str, AgentSpec] = {}
        self._agents: dict[str, CompiledStateGraph] = {}  # mode="react" ReAct graphs
        self._tool_agents: dict[str, BaseTool] = {}  # mode="tool_call" primary tools

    def register(self, spec: AgentSpec) -> None:
        """Register an agent specification."""
        self._specs[spec.id] = spec
        logger.info(f"Registered agent: {spec.id} ({spec.display_name}, mode={spec.mode})")

    def build_agents(self, llm: ChatOpenAI) -> dict[str, CompiledStateGraph]:
        """Build all registered agents according to their execution mode.

        mode="tool_call" agents are wired to their primary tool (no ReAct graph);
        mode="react" agents are compiled as create_react_agent loops.
        """
        for agent_id, spec in self._specs.items():
            if not spec.tools:
                logger.warning(f"Agent '{agent_id}' has no tools, skipping build")
                continue

            if spec.mode == "tool_call":
                tool = next((t for t in spec.tools if t.name == spec.primary_tool), None)
                if tool is None:
                    logger.error(
                        f"Agent '{agent_id}' mode=tool_call but primary_tool "
                        f"'{spec.primary_tool}' not found in its tools; skipping build"
                    )
                    continue
                self._tool_agents[agent_id] = tool
                logger.info(f"Built agent: {agent_id} (tool_call → {tool.name})")
                continue

            # Auto-generate a prompt for agents registered without one (e.g. MCP agents).
            prompt = spec.system_prompt
            if not prompt:
                from security_intel.prompts.agent import render_agent_system_prompt
                prompt = render_agent_system_prompt(
                    spec.display_name, spec.description, spec.capabilities
                )

            agent = create_react_agent(
                model=llm,
                tools=spec.tools,
                prompt=prompt,
            )
            self._agents[agent_id] = agent
            logger.info(f"Built agent: {agent_id} (react, {len(spec.tools)} tools)")

        return self._agents

    def _is_built(self, agent_id: str) -> bool:
        return agent_id in self._agents or agent_id in self._tool_agents

    def get_agent(self, agent_id: str) -> CompiledStateGraph | None:
        """Get a built ReAct agent by ID (mode=react only)."""
        return self._agents.get(agent_id)

    def get_tool_agent(self, agent_id: str) -> BaseTool | None:
        """Get a tool_call agent's primary tool by ID (mode=tool_call only)."""
        return self._tool_agents.get(agent_id)

    def get_mode(self, agent_id: str) -> str:
        """Execution mode of a built agent ('react' | 'tool_call')."""
        spec = self._specs.get(agent_id)
        return spec.mode if spec else "react"

    def get_spec(self, agent_id: str) -> AgentSpec | None:
        """Get agent specification by ID."""
        return self._specs.get(agent_id)

    @property
    def agent_ids(self) -> list[str]:
        # Union of both execution modes, in registration order.
        return [aid for aid in self._specs if self._is_built(aid)]

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
            if not self._is_built(agent_id):
                continue

            # Create a describe tool for this agent
            desc_tool = _make_describe_tool(agent_id, spec)
            planner_tools.append(desc_tool)

        # The plan creation tool
        planner_tools.append(_make_plan_tool(self.agent_ids))
        return planner_tools

    def build_agent_catalog(self) -> str:
        """Rich, single source of truth for agent routing context.

        Full description + top capabilities per built agent. Used by BOTH the
        router (classify) and the planner so routing decisions are never made on
        truncated agent info. Regenerate whenever the built-agent set changes.
        """
        lines = []
        for agent_id, spec in self._specs.items():
            if not self._is_built(agent_id):
                continue
            caps = "; ".join(spec.capabilities[:4])
            desc = " ".join((spec.description or "").split())  # collapse whitespace
            line = f"- {agent_id}: {desc}"
            if caps:
                line += f" Capabilities: {caps}"
            lines.append(line)
        return "\n".join(lines)

    def build_planner_system_prompt(self) -> str:
        """Auto-generate planner system prompt from registered agents."""
        return PLANNER_SYSTEM_TEMPLATE.format(agents_block=self.build_agent_catalog())


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

    desc = f"""Create the execution plan for sub-agents.

        Args:
            steps: List of plan steps. Each step is a dict with:
                - agent: One of [{agents_str}]
                - task: The specific sub-question for that agent (self-contained)
                - depends_on: List of step indices (0-based) that must complete first
            synthesis_goal: How the synthesizer should combine the findings.
        """

    @tool(description=desc)
    def create_execution_plan(steps: list[dict], synthesis_goal: str) -> str:

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
