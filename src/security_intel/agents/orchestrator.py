"""Production orchestrator — LangGraph StateGraph coordinating the full agent pipeline.

Architecture:
    START → input_guard → load_context → classify → plan → validate_plan → dispatch → synthesize → output_guard → persist → END

Production features:
- Conversation context fed to planner (multi-turn reasoning)
- Complexity classification routes to appropriate model tier
- Timeouts on all LLM/agent calls (no hung requests)
- Recursion limits on sub-agents (no infinite tool loops)
- Structured error recovery (fallback, not crash)
- Agent registry integration (dynamic agent discovery)
- Session persistence with rolling summarization
"""

import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from security_intel.state.schemas import OrchestratorState, AgentResult, ExecutionPlan, PlanStep
from security_intel.llm.provider import LaneRouter
from security_intel.memory.conversations import ConversationStore, ChatSession
from security_intel.memory.summarizer import RollingSummarizer
from security_intel.agents.registry import AgentRegistry
from security_intel.observability.logging import get_logger, set_trace_context

logger = get_logger("orchestrator")

# Timeouts (seconds)
PLANNER_TIMEOUT = 30
SUB_AGENT_TIMEOUT = 60
SYNTHESIS_TIMEOUT = 45
CLASSIFY_TIMEOUT = 10

# LangGraph recursion limit for sub-agents (prevents infinite tool loops)
AGENT_RECURSION_LIMIT = 15

COMPLEXITY_PROMPT = """Classify query complexity for a security intelligence system.

SIMPLE: single domain, direct lookup, one agent (e.g., "What is CVE-2024-1234?")
COMPLEX: multi-domain, cross-referencing, multi-step analysis (e.g., "Compare our exposed assets against recent threat reports")

Question: {question}

Context from prior conversation:
{context}

Respond ONLY: SIMPLE or COMPLEX"""

SYNTHESIS_PROMPT = """You are a security intelligence analyst synthesizing findings from specialist agents.

Rules:
1. Combine findings into clear, actionable answer
2. Cite sources: [Report: title] or [EASM: finding]
3. Highlight CRITICAL items first
4. Note conflicts between findings
5. Be concise for simple queries, structured for complex ones
6. For security professionals — no hand-holding, be precise"""


def build_orchestrator(
    lane_router: LaneRouter,
    registry: AgentRegistry,
    conversations: ConversationStore | None = None,
    summarizer: RollingSummarizer | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build production orchestrator as LangGraph StateGraph."""

    # Build planner from registry (auto-discovers agents)
    planner_tools = registry.build_planner_tools()
    planner_prompt = registry.build_planner_system_prompt()

    from langgraph.prebuilt import create_react_agent
    planner = create_react_agent(
        model=lane_router.fast,
        tools=planner_tools,
        prompt=planner_prompt,
    )

    # -------------------------------------------------------------------------
    # Graph Nodes
    # -------------------------------------------------------------------------

    async def input_guardrail_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Injection/safety check. Blocks malicious input."""
        from security_intel.security.guardrails import input_guardrail_node as guard
        result = await guard(state, config)
        if result.get("blocked"):
            logger.warning("Input blocked", extra={"extra_data": {
                "reason": result.get("block_reason"),
                "query": state["user_query"][:100],
            }})
        return result

    async def load_context_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Load conversation history + rolling summary for multi-turn context."""
        if not conversations:
            return {}

        org_id = config["configurable"].get("org_id", "")
        session_id = config["configurable"].get("thread_id", "")
        if not org_id or not session_id:
            return {}

        try:
            session = await conversations.get_session(org_id, session_id)
            if not session:
                return {}

            history = await conversations.get_messages(org_id, session_id, limit=20)
            context_messages = []

            if session.summary:
                context_messages.append(
                    SystemMessage(content=f"Conversation summary: {session.summary}")
                )
            for msg in history[-10:]:  # Last 10 messages as direct context
                if msg.role == "user":
                    context_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    context_messages.append(AIMessage(content=msg.content))

            return {"messages": context_messages}
        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            return {}

    async def classify_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Classify complexity → route to STANDARD or DEEP reasoning."""
        question = state["user_query"]
        context = _summarize_context(state.get("messages", []))

        try:
            response = await asyncio.wait_for(
                lane_router.fast.ainvoke([
                    HumanMessage(content=COMPLEXITY_PROMPT.format(
                        question=question, context=context or "(new conversation)"
                    ))
                ]),
                timeout=CLASSIFY_TIMEOUT,
            )
            is_complex = "complex" in response.content.strip().lower()
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Classification failed ({e}), defaulting to SIMPLE")
            is_complex = False

        logger.info(f"Query classified: {'COMPLEX' if is_complex else 'SIMPLE'}")
        return {"is_complex": is_complex}

    async def plan_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Invoke planner agent with full conversation context."""
        # Give planner the conversation context + current query
        planner_messages = []
        prior = state.get("messages", [])
        if prior:
            # Include last few turns so planner understands context
            for msg in prior[-6:]:
                planner_messages.append(msg)

        planner_messages.append(HumanMessage(content=state["user_query"]))

        try:
            result = await asyncio.wait_for(
                planner.ainvoke(
                    {"messages": planner_messages},
                    config=RunnableConfig(recursion_limit=10),
                ),
                timeout=PLANNER_TIMEOUT,
            )
            plan = _extract_plan(result["messages"], registry.agent_ids)
            logger.info(f"Plan: {len(plan['steps'])} steps → {[s['agent'] for s in plan['steps']]}")
            return {"plan": plan}
        except asyncio.TimeoutError:
            logger.error("Planner timed out, falling back to default plan")
            return {"plan": _default_plan(state["user_query"], registry.agent_ids)}
        except Exception as e:
            logger.error(f"Planner failed: {e}", exc_info=True)
            return {"plan": _default_plan(state["user_query"], registry.agent_ids)}

    async def validate_plan_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Validate plan DAG structure — catch cycles, invalid agents."""
        plan = state.get("plan")
        if not plan or not plan.get("steps"):
            return {"plan": _default_plan(state["user_query"], registry.agent_ids)}

        valid_agents = set(registry.agent_ids)
        validated_steps = []

        for step in plan["steps"]:
            if step["agent"] not in valid_agents:
                logger.warning(f"Plan references unknown agent '{step['agent']}', skipping")
                continue
            # Validate depends_on indices
            deps = [d for d in step.get("depends_on", []) if 0 <= d < len(plan["steps"])]
            validated_steps.append(PlanStep(
                agent=step["agent"],
                task=step["task"],
                depends_on=deps,
            ))

        if not validated_steps:
            return {"plan": _default_plan(state["user_query"], registry.agent_ids)}

        # Cycle detection
        if _has_cycle(validated_steps):
            logger.warning("Plan has dependency cycle, removing all dependencies")
            for step in validated_steps:
                step["depends_on"] = []

        return {"plan": ExecutionPlan(steps=validated_steps, synthesis_goal=plan["synthesis_goal"])}

    async def dispatch_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Dispatch to sub-agents in topological order with timeouts."""
        plan = state["plan"]
        steps = plan["steps"]
        results: list[AgentResult] = []
        completed: dict[int, str] = {}

        batches = _topological_sort(steps)
        org_id = config["configurable"].get("org_id", "")

        for batch in batches:
            tasks = []
            for idx in batch:
                step = steps[idx]
                task_content = step["task"]

                # Inject prior step context for dependent tasks
                if step.get("depends_on"):
                    prior_context = [completed[d] for d in step["depends_on"] if d in completed]
                    if prior_context:
                        task_content += "\n\nContext from prior analysis:\n" + "\n---\n".join(prior_context)

                agent = registry.get_agent(step["agent"])
                if not agent:
                    tasks.append(_make_error_result(step["agent"], f"Agent '{step['agent']}' not available"))
                    continue

                tasks.append(_invoke_agent(agent, task_content, config, step["agent"]))

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    agent_result = AgentResult(
                        agent_id=steps[idx]["agent"],
                        findings=f"Agent failed: {type(result).__name__}: {result}",
                        citations=[],
                        tool_calls=[],
                    )
                    logger.error(f"Agent {steps[idx]['agent']} failed", exc_info=result)
                else:
                    agent_result = result

                results.append(agent_result)
                completed[idx] = agent_result["findings"]

        return {"agent_results": results}

    async def synthesize_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Synthesize findings — uses DEEP model for complex queries."""
        agent_results = state.get("agent_results", [])
        plan = state.get("plan")
        user_query = state["user_query"]
        is_complex = state.get("is_complex", False)

        if not agent_results:
            return {
                "final_answer": "I couldn't find relevant information to answer your question. "
                               "Please try rephrasing or ask about a specific topic.",
                "citations": [],
            }

        # Single agent, simple query → pass through directly (no synthesis LLM call)
        if len(agent_results) == 1 and not is_complex:
            findings = agent_results[0]["findings"]
            if not findings.startswith("Agent failed:"):
                return {
                    "final_answer": findings,
                    "citations": agent_results[0]["citations"],
                    "messages": [AIMessage(content=findings)],
                }

        # Multi-agent or complex → synthesize with appropriate model
        llm = lane_router.deep if is_complex else lane_router.standard

        findings_text = ""
        all_citations = []
        for r in agent_results:
            findings_text += f"\n\n### {r['agent_id'].upper()} Agent Findings:\n{r['findings']}"
            all_citations.extend(r["citations"])

        synthesis_goal = plan.get("synthesis_goal", "Combine into clear answer.") if plan else ""

        prompt = (
            f"User question: {user_query}\n"
            f"Synthesis goal: {synthesis_goal}\n"
            f"Agent findings:{findings_text}"
        )

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(
                    [SystemMessage(content=SYNTHESIS_PROMPT), HumanMessage(content=prompt)],
                    config=config,
                ),
                timeout=SYNTHESIS_TIMEOUT,
            )
            answer = response.content
        except asyncio.TimeoutError:
            logger.error("Synthesis timed out, returning raw findings")
            answer = "Here are the findings from the analysis:\n" + findings_text
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            answer = "Here are the findings from the analysis:\n" + findings_text

        return {
            "final_answer": answer,
            "citations": all_citations,
            "messages": [AIMessage(content=answer)],
        }

    async def output_guardrail_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """PII/groundedness check on output."""
        from security_intel.security.guardrails import output_guardrail_node as guard
        return await guard(state, config)

    async def persist_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Persist turn to Postgres + roll summary if needed."""
        if not conversations:
            return {}

        org_id = config["configurable"].get("org_id", "")
        session_id = config["configurable"].get("thread_id", "")
        user_id = config["configurable"].get("user_id", "")
        if not org_id or not session_id:
            return {}

        try:
            session = await conversations.get_session(org_id, session_id)
            if not session:
                session = await conversations.create_session(org_id, user_id, state["user_query"][:60])
                session_id = session.id

            await conversations.append_message(org_id, session_id, "user", state["user_query"])

            agents_used = [r["agent_id"] for r in state.get("agent_results", [])]
            await conversations.append_message(
                org_id, session_id, "assistant", state.get("final_answer", ""),
                citations=state.get("citations", []),
                meta={"agents_used": agents_used, "is_complex": state.get("is_complex", False)},
            )

            if summarizer:
                session = await conversations.get_session(org_id, session_id)
                if session:
                    await summarizer.maybe_summarize(org_id, session)

        except Exception as e:
            logger.error(f"Persist failed: {e}", exc_info=True)

        return {}

    # -------------------------------------------------------------------------
    # Build Graph
    # -------------------------------------------------------------------------

    graph = StateGraph(OrchestratorState)

    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("load_context", load_context_node)
    graph.add_node("classify", classify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("output_guardrail", output_guardrail_node)
    graph.add_node("persist", persist_node)

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        lambda s: "blocked" if s.get("blocked") else "continue",
        {"blocked": END, "continue": "load_context"},
    )
    graph.add_edge("load_context", "classify")
    graph.add_edge("classify", "plan")
    graph.add_edge("plan", "validate_plan")
    graph.add_edge("validate_plan", "dispatch")
    graph.add_edge("dispatch", "synthesize")
    graph.add_edge("synthesize", "output_guardrail")
    graph.add_edge("output_guardrail", "persist")
    graph.add_edge("persist", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(f"Orchestrator built: {len(registry.agent_ids)} agents registered")
    return compiled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _invoke_agent(
    agent: CompiledStateGraph,
    task: str,
    config: RunnableConfig,
    agent_id: str,
) -> AgentResult:
    """Invoke a LangGraph sub-agent with timeout and recursion limit."""
    agent_config = RunnableConfig(
        configurable=config["configurable"],
        recursion_limit=AGENT_RECURSION_LIMIT,
    )

    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [HumanMessage(content=task)]}, config=agent_config),
            timeout=SUB_AGENT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return AgentResult(
            agent_id=agent_id,
            findings=f"Agent '{agent_id}' timed out after {SUB_AGENT_TIMEOUT}s. The query may be too broad.",
            citations=[],
            tool_calls=[],
        )

    messages = result.get("messages", [])
    final_content = ""
    tool_calls_log = []

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append({"name": tc["name"], "args": tc.get("args", {})})
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            final_content = msg.content

    if not final_content and messages:
        final_content = getattr(messages[-1], "content", "") or "No findings."

    return AgentResult(
        agent_id=agent_id,
        findings=final_content,
        citations=[],
        tool_calls=tool_calls_log,
    )


async def _make_error_result(agent_id: str, error_msg: str) -> AgentResult:
    """Create an error result without invoking an agent."""
    return AgentResult(agent_id=agent_id, findings=error_msg, citations=[], tool_calls=[])


def _extract_plan(messages: list, valid_agents: list[str]) -> ExecutionPlan:
    """Extract ExecutionPlan from planner's create_execution_plan tool call."""
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "create_execution_plan":
                    args = tc["args"]
                    steps = []
                    for s in args.get("steps", []):
                        steps.append(PlanStep(
                            agent=s["agent"],
                            task=s["task"],
                            depends_on=s.get("depends_on", []),
                        ))
                    return ExecutionPlan(
                        steps=steps,
                        synthesis_goal=args.get("synthesis_goal", "Combine findings."),
                    )

    # Planner didn't call the tool — use first available agent
    fallback_agent = valid_agents[0] if valid_agents else "reports"
    return ExecutionPlan(
        steps=[PlanStep(agent=fallback_agent, task="Answer the user's question", depends_on=[])],
        synthesis_goal="Return findings directly.",
    )


def _default_plan(query: str, valid_agents: list[str]) -> ExecutionPlan:
    """Fallback plan when planner fails."""
    agent = valid_agents[0] if valid_agents else "reports"
    return ExecutionPlan(
        steps=[PlanStep(agent=agent, task=query, depends_on=[])],
        synthesis_goal="Return findings directly.",
    )


def _has_cycle(steps: list[PlanStep]) -> bool:
    """Detect cycles in plan dependencies."""
    n = len(steps)
    visited = [0] * n  # 0=unvisited, 1=in-progress, 2=done

    def dfs(i: int) -> bool:
        if visited[i] == 1:
            return True
        if visited[i] == 2:
            return False
        visited[i] = 1
        for dep in steps[i].get("depends_on", []):
            if 0 <= dep < n and dfs(dep):
                return True
        visited[i] = 2
        return False

    return any(dfs(i) for i in range(n) if visited[i] == 0)


def _topological_sort(steps: list[PlanStep]) -> list[list[int]]:
    """Group steps into parallel batches respecting dependencies."""
    n = len(steps)
    if n == 0:
        return []

    batches = []
    done = set()

    while len(done) < n:
        batch = [
            i for i in range(n)
            if i not in done and all(d in done for d in steps[i].get("depends_on", []))
        ]
        if not batch:
            # Shouldn't happen after cycle detection, but safety net
            batches.append([i for i in range(n) if i not in done])
            break
        batches.append(batch)
        done.update(batch)

    return batches


def _summarize_context(messages: list) -> str:
    """Create brief context summary for the classifier."""
    if not messages:
        return ""
    parts = []
    for msg in messages[-4:]:
        content = getattr(msg, "content", "")
        if content:
            parts.append(f"{type(msg).__name__}: {content[:100]}")
    return "\n".join(parts)
