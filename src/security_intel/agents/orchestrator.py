"""Production orchestrator — LangGraph StateGraph coordinating the full agent pipeline.

Architecture:
    START → security_gate → load_context → classify → plan → validate_plan → dispatch → synthesize → output_guard → persist → END

Key design decisions:
- Security checks run in parallel with early cancellation (fast-fail on threats)
- Sub-agents dispatched in parallel batches (topological order)
- Orchestrator has full conversational context for intelligent planning
- Friendly, knowledgeable persona that understands security intelligence domain
- Optimized for low time-to-first-token via parallel execution
"""

import asyncio
import json

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
from security_intel.prompts.orchestrator import (
    ORCHESTRATOR_PERSONA,
    ROUTER_PROMPT,
    CHITCHAT_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTH_FALLBACK_MSG,
)

logger = get_logger("orchestrator")

# Timeouts (seconds)
PLANNER_TIMEOUT = 30
SUB_AGENT_TIMEOUT = 60
SYNTHESIS_TIMEOUT = 45
CLASSIFY_TIMEOUT = 10
SECURITY_TIMEOUT = 8

# LangGraph recursion limit for sub-agents
AGENT_RECURSION_LIMIT = 15

def build_orchestrator(
    lane_router: LaneRouter,
    registry: AgentRegistry,
    conversations: ConversationStore | None = None,
    summarizer: RollingSummarizer | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    query_enricher=None,
) -> CompiledStateGraph:
    """Build production orchestrator as LangGraph StateGraph."""

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

    async def security_gate_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Tiered security checks — regex-only for short queries, full LLM check for longer ones.

        Optimization: queries under 200 chars skip the LLM injection check (~2-3s savings)
        since regex patterns catch known attacks and short queries have limited attack surface.
        LLM check still runs for longer queries where sophisticated attacks hide.
        """
        from security_intel.security.guardrails import input_guardrail_node

        query = state["user_query"]
        use_llm = len(query) > 200
        llm = lane_router.fast if use_llm else None

        try:
            result = await asyncio.wait_for(
                input_guardrail_node(state, config, llm=llm),
                timeout=SECURITY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Security checks timed out — allowing request (fail-open for availability)")
            return {"blocked": False, "block_reason": ""}

        if result.get("blocked"):
            logger.warning("Input blocked", extra={"extra_data": {
                "reason": result.get("block_reason"),
                "query": query[:100],
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
            for msg in history[-10:]:
                if msg.role == "user":
                    context_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    context_messages.append(AIMessage(content=msg.content))

            return {"messages": context_messages}
        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            return {}

    async def classify_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Unified router that classifies AND plans for simple queries.

        For SIMPLE queries, the router generates the agent task inline — eliminating
        the separate planner LLM call entirely. Only COMPLEX queries proceed to the
        full planner for multi-agent decomposition.
        """
        question = state["user_query"]
        context = _summarize_context(state.get("messages", []))
        agents_str = ", ".join(
            f"{aid} ({registry.get_spec(aid).description[:60]})"
            for aid in registry.agent_ids
        )

        try:
            response = await asyncio.wait_for(
                lane_router.fast.ainvoke([
                    HumanMessage(content=ROUTER_PROMPT.format(
                        question=question,
                        context=context or "(new conversation)",
                        agents=agents_str,
                    ))
                ]),
                timeout=CLASSIFY_TIMEOUT,
            )
            parsed = _parse_router_response(response.content)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Router failed ({e}), defaulting to SIMPLE")
            parsed = {"action": "SIMPLE"}

        action = parsed.get("action", "SIMPLE").upper()

        if action == "DIRECT":
            logger.info("Query routed: DIRECT (no agents needed)")
            return {
                "is_complex": False,
                "is_chitchat": True,
                "direct_response": parsed.get("response", ""),
            }

        if action == "SIMPLE" and parsed.get("agent") and parsed.get("task"):
            agent_id = parsed["agent"]
            if agent_id in registry.agent_ids:
                logger.info(f"Query routed: SIMPLE → {agent_id} (planner bypassed)")
                plan = ExecutionPlan(
                    steps=[PlanStep(agent=agent_id, task=parsed["task"], depends_on=[])],
                    synthesis_goal="Return findings directly.",
                )
                return {"is_complex": False, "is_chitchat": False, "plan": plan}
            else:
                logger.warning(f"Router picked unknown agent '{agent_id}', falling through to planner")

        is_complex = action == "COMPLEX"
        logger.info(f"Query routed: {action} → planner")
        return {"is_complex": is_complex, "is_chitchat": False}

    async def plan_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Invoke planner agent with full conversation context."""
        planner_messages = []
        prior = state.get("messages", [])
        if prior:
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
        """Validate plan DAG — catch cycles, invalid agents."""
        plan = state.get("plan")
        if not plan or not plan.get("steps"):
            return {"plan": _default_plan(state["user_query"], registry.agent_ids)}

        valid_agents = set(registry.agent_ids)
        validated_steps = []

        for step in plan["steps"]:
            if step["agent"] not in valid_agents:
                logger.warning(f"Plan references unknown agent '{step['agent']}', skipping")
                continue
            deps = [d for d in step.get("depends_on", []) if 0 <= d < len(plan["steps"])]
            validated_steps.append(PlanStep(
                agent=step["agent"],
                task=step["task"],
                depends_on=deps,
            ))

        if not validated_steps:
            return {"plan": _default_plan(state["user_query"], registry.agent_ids)}

        if _has_cycle(validated_steps):
            logger.warning("Plan has dependency cycle, removing all dependencies")
            for step in validated_steps:
                step["depends_on"] = []

        return {"plan": ExecutionPlan(steps=validated_steps, synthesis_goal=plan["synthesis_goal"])}

    async def dispatch_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Dispatch sub-agents in parallel batches (topological order).

        Independent agents run concurrently for minimum wall-clock time.
        Dependent agents wait only for their specific dependencies.

        When query_enricher is available and query is complex, enriches agent tasks
        with expanded search hints so agents start with better context.
        """
        plan = state["plan"]
        steps = plan["steps"]
        results: list[AgentResult] = []
        completed: dict[int, str] = {}

        batches = _topological_sort(steps)

        for batch in batches:
            tasks = []
            for idx in batch:
                step = steps[idx]
                task_content = step["task"]

                if step.get("depends_on"):
                    prior_context = [completed[d] for d in step["depends_on"] if d in completed]
                    if prior_context:
                        task_content += "\n\nContext from prior analysis:\n" + "\n---\n".join(prior_context)

                if query_enricher and state.get("is_complex"):
                    task_content = await _enrich_agent_task(query_enricher, task_content)

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
        """Synthesize findings with conversational persona."""
        agent_results = state.get("agent_results", [])
        plan = state.get("plan")
        user_query = state["user_query"]
        is_complex = state.get("is_complex", False)

        # Drop error/timeout results — their raw diagnostics must never reach the user.
        real_results = [r for r in agent_results if not _is_error_finding(r["findings"])]

        if not real_results:
            friendly_empty = (
                "I wasn't able to retrieve results for that just now. This can happen "
                "when a query is very broad or a source is slow to respond. Could you try:\n"
                "- Being more specific (e.g., a CVE ID, report ID, asset name, or date range)\n"
                "- Rephrasing your question\n"
                "- Asking again in a moment"
            )
            return {"final_answer": friendly_empty, "citations": []}

        # NOTE: previously single-agent simple queries returned findings directly
        # here (no LLM call). That path emitted zero on_chat_model_stream events,
        # so the UI never streamed — the answer only arrived in the final `done`.
        # Always route through the synthesis LLM so tokens stream from this node.
        llm = lane_router.deep if is_complex else lane_router.standard

        findings_text = ""
        all_citations = []
        for r in real_results:
            findings_text += f"\n\n### {r['agent_id'].upper()} Agent Findings:\n{r['findings']}"
            all_citations.extend(r["citations"])

        synthesis_goal = plan.get("synthesis_goal", "Combine into clear answer.") if plan else ""

        context_summary = _summarize_context(state.get("messages", []))
        context_block = f"\nPrior conversation context:\n{context_summary}\n" if context_summary else ""

        prompt = (
            f"User question: {user_query}\n"
            f"{context_block}"
            f"Synthesis goal: {synthesis_goal}\n"
            f"Agent findings:{findings_text}"
        )

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(
                    [
                        SystemMessage(content=SYNTHESIS_PROMPT.format(persona=ORCHESTRATOR_PERSONA)),
                        HumanMessage(content=prompt),
                    ],
                    config=config,
                ),
                timeout=SYNTHESIS_TIMEOUT,
            )
            answer = response.content
        except asyncio.TimeoutError:
            logger.error("Synthesis timed out")
            answer = SYNTH_FALLBACK_MSG
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            answer = SYNTH_FALLBACK_MSG

        return {
            "final_answer": answer,
            "citations": all_citations,
            "messages": [AIMessage(content=answer)],
        }

    async def context_and_classify_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Run load_context and classify in parallel — saves ~1-2s on every request."""
        context_task = asyncio.create_task(load_context_node(state, config))
        classify_task = asyncio.create_task(classify_node(state, config))

        ctx_result, cls_result = await asyncio.gather(context_task, classify_task)

        merged = {}
        merged.update(ctx_result)
        merged.update(cls_result)
        return merged

    async def chitchat_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """Generate the DIRECT answer with a streaming LLM call.

        Runs an actual LLM invocation (not a pass-through) so token events fire
        from this node — DIRECT/off-topic answers stream to the UI like every
        other response. The router's inline draft is used only as a fallback.
        """
        user_query = state["user_query"]
        fallback = state.get("direct_response", "")

        try:
            response = await asyncio.wait_for(
                lane_router.standard.ainvoke(
                    [
                        SystemMessage(content=CHITCHAT_PROMPT.format(persona=ORCHESTRATOR_PERSONA)),
                        HumanMessage(content=user_query),
                    ],
                    config=config,
                ),
                timeout=SYNTHESIS_TIMEOUT,
            )
            answer = response.content
        except Exception as e:
            logger.warning(f"Chitchat generation failed ({e}), using router draft")
            answer = fallback or (
                "Hello! I'm your Security Intelligence Assistant. Ask me about "
                "threats, CVEs, attack surface, or security reports."
            )

        return {
            "final_answer": answer,
            "citations": [],
            "agent_results": [],
            "messages": [AIMessage(content=answer)],
        }

    async def output_guardrail_node(state: OrchestratorState, config: RunnableConfig) -> dict:
        """PII redaction on output via Presidio."""
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

    graph.add_node("security_gate", security_gate_node)
    graph.add_node("context_and_classify", context_and_classify_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("plan", plan_node)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("output_guardrail", output_guardrail_node)
    graph.add_node("persist", persist_node)

    graph.add_edge(START, "security_gate")
    graph.add_conditional_edges(
        "security_gate",
        lambda s: "blocked" if s.get("blocked") else "continue",
        {"blocked": END, "continue": "context_and_classify"},
    )
    graph.add_conditional_edges(
        "context_and_classify",
        lambda s: "chitchat" if s.get("is_chitchat") else ("dispatch" if s.get("plan") else "plan"),
        {"chitchat": "chitchat", "plan": "plan", "dispatch": "dispatch"},
    )
    graph.add_edge("chitchat", "persist")
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
    max_retries: int = 1,
) -> AgentResult:
    """Invoke a LangGraph sub-agent with timeout, recursion limit, and retry on transient errors."""
    agent_config = RunnableConfig(
        configurable=config["configurable"],
        recursion_limit=AGENT_RECURSION_LIMIT,
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": [HumanMessage(content=task)]}, config=agent_config),
                timeout=SUB_AGENT_TIMEOUT,
            )
            break
        except asyncio.TimeoutError:
            return AgentResult(
                agent_id=agent_id,
                findings=f"Agent '{agent_id}' timed out after {SUB_AGENT_TIMEOUT}s. The query may be too broad.",
                citations=[],
                tool_calls=[],
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(f"Agent '{agent_id}' attempt {attempt+1} failed: {e}, retrying...")
                await asyncio.sleep(0.5)
            else:
                return AgentResult(
                    agent_id=agent_id,
                    findings=f"Agent '{agent_id}' failed after {max_retries+1} attempts: {e}",
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


_ERROR_FINDING_MARKERS = (
    "Agent failed",
    "timed out after",
    "not available",
    "All connection attempts failed",
)


def _is_error_finding(findings: str) -> bool:
    """True if an agent result is an internal error/timeout, not real content.

    Used to keep raw diagnostics out of the synthesized, user-facing answer.
    """
    if not findings:
        return True
    head = findings[:200]
    return any(m in head for m in _ERROR_FINDING_MARKERS)


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
    visited = [0] * n

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
            batches.append([i for i in range(n) if i not in done])
            break
        batches.append(batch)
        done.update(batch)

    return batches


def _parse_router_response(content: str) -> dict:
    """Extract JSON from router LLM response, handling markdown fences."""
    text = content.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"action": "SIMPLE"}


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


async def _enrich_agent_task(enricher, task: str) -> str:
    """Enrich a complex agent task with search hints from query expansion.

    For complex multi-domain queries, provides the agent with alternative
    search angles so it can make more targeted tool calls.
    """
    try:
        enriched = await enricher.enrich(task)
        if len(enriched.search_queries) <= 1:
            return task

        hints = "\n".join(f"  - {q}" for q in enriched.search_queries[1:])
        return (
            f"{task}\n\n"
            f"Search optimization hints (alternative angles to try):\n{hints}"
        )
    except Exception as e:
        logger.debug(f"Task enrichment failed ({e}), using original task")
        return task
