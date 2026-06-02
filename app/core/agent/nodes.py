"""The agent graph NODES (engine-agnostic) — the actual steps of a chat turn.

This is where the reasoning lives. Each function here is ONE node in the graph
described by graph.py. Read them top to bottom and you have read the whole agent.

The flow (same order both engines run):

    input_guardrail -> route -> gather_context -> answer -> output_guardrail

  * input_guardrail — screen the user's question (redact secrets, block prompt
    injection / unsafe topics). Can short-circuit the whole turn to END.
  * route           — the SUPERVISOR picks which capability module(s) should
    answer, using only each module's manifest routing hints.
  * gather_context  — dispatch ONE specialist per routed module, IN PARALLEL.
    Each specialist is scoped to its OWN module's tools (so tool schemas never
    pile up in one context — this is the key to scaling to hundreds of tools).
    Their findings are merged, ranked, and capped into a numbered context block.
  * answer          — the single SYNTHESIZE step: the LLM joins the findings and
    writes the answer, citing sources as [1], [2]; streams tokens if asked.
  * output_guardrail— verify the answer is grounded in the context and leaks no
    PII before it goes out.

Contract every node obeys (the graph.py rule): take ``(state, ctx)`` and return
a dict of ONLY the keys it changed. The engine merges that into the shared
ChatState. ``ctx`` (AgentContext) carries the services + the streaming sink.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.agent.graph import END, StateGraph
from app.core.agent.specialist import build_specialist, relevance_rank
from app.core.agent.state import (
    N_ANSWER,
    N_GATHER,
    N_INPUT_GUARD,
    N_OUTPUT_GUARD,
    N_PLAN,
    N_PLAN_DISPATCH,
    N_REPLAN_GATE,
    N_ROUTE,
    AgentContext,
    ChatState,
)
from app.core.contracts import Chunk
from app.core.llm.base import CONTEXT_MARKER, ChatMessage, Lane

# After merging every specialist's findings we keep at most this many chunks for
# the final prompt. A hard cap is what keeps the LLM context bounded no matter
# how many modules answered — cost and latency stay predictable.
MAX_CONTEXT_ENTRIES = 8

# The base "system prompt" — the standing instructions given to the answer LLM.
# Note the two rules that make this a RAG system and not a chatbot: answer ONLY
# from the retrieved context, and refuse (don't guess) when the context lacks
# the answer. This is the anti-hallucination contract, enforced again at output.
BASE_SYSTEM = (
    "You are a security-intelligence analyst assistant. Answer the user's "
    "question USING ONLY the retrieved context below. Cite every claim with its "
    "source marker like [1]. If the context does not contain the answer, say you "
    "don't have enough grounded information — never guess. Be concise and precise."
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def build_answer_messages(state: ChatState, ctx: AgentContext) -> list[ChatMessage]:
    """Assemble the exact message list sent to the answer LLM.

    The ORDER and ROLE of these messages matters a lot for answer quality —
    this is "prompt engineering" made concrete:

      1. system: persona + base rules (optionally the routed module's own prompt)
      2. system: the rolling conversation summary (if any)
      3. user/assistant: the last few turns of real history (for follow-ups)
      4. system: the RETRIEVED CONTEXT, in its OWN message, clearly marked
      5. user: the bare question

    Keeping the context in its own system message (step 4), separate from the
    user's question (step 5), is deliberate: it stops the model from treating
    retrieved text as if the *user* said it, and keeps "instructions vs data"
    cleanly separated (a basic prompt-injection hygiene measure).
    """
    persona = BASE_SYSTEM
    routed = state.get("route_modules") or []
    # If a routed module ships its own system prompt (tone/domain expertise),
    # prepend the FIRST one for flavor. Modules declare this via their manifest.
    for mid in routed:
        mod = ctx.registry.module(mid)
        if mod and mod.prompt_text:
            persona = f"{BASE_SYSTEM}\n\n{mod.prompt_text}"
            break

    messages = [ChatMessage(role="system", content=persona)]
    if state.get("summary"):
        messages.append(ChatMessage(role="system", content=f"Conversation summary so far: {state['summary']}"))
    # Only the last N turns (settings.answer_history_turns) — recent context
    # matters most, older context is covered by the rolling summary above. This is
    # the ChatGPT-style "last-N messages as context" window; it bounds the prompt
    # on long conversations while keeping the thread coherent for follow-ups.
    n_turns = getattr(ctx.settings, "answer_history_turns", 6)
    for turn in (state.get("history") or [])[-n_turns:]:
        role = turn.get("role", "user")
        if role in ("user", "assistant"):
            messages.append(ChatMessage(role=role, content=turn.get("content", "")))

    context_block = state.get("context_block") or "(no sources retrieved)"
    # The context goes in its OWN system message (see the docstring). CONTEXT_MARKER
    # is a constant label so the prompt format is consistent and testable.
    messages.append(ChatMessage(
        role="system",
        content=f"{CONTEXT_MARKER} (cite as [n]; if it lacks the answer, say so):\n{context_block}",
    ))
    # The user message is just the (sanitized) question — nothing else.
    messages.append(ChatMessage(role="user", content=state.get("safe_question") or state.get("question", "")))
    return messages


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
async def input_guardrail_node(state: ChatState, ctx: AgentContext) -> dict:
    """NODE 1 — screen the incoming question BEFORE the agent does anything.

    Runs the input guardrail spine: redact secrets, detect prompt injection,
    check topic safety. If it blocks, we set ``blocked=True`` and stash a safe
    refusal as the answer; the graph's conditional edge then routes straight to
    END (see build_report_graph), so no routing/retrieval/LLM call ever happens.
    """
    await ctx.fire("status", stage="input_guardrail")     # tell the UI what stage we're in
    res = await ctx.input_guard.run(state.get("question", ""), ctx.sc)
    if res.blocked:
        await ctx.fire("status", stage="blocked", reason=(res.reasons or ["blocked"])[0])
        return {"blocked": True, "block_reason": (res.reasons or ["blocked"])[0],
                "answer": res.text, "safe_question": state.get("question", "")}
    # Not blocked: ``res.text`` is the cleaned question we use downstream.
    return {"blocked": False, "safe_question": res.text}


async def route_node(state: ChatState, ctx: AgentContext) -> dict:
    """NODE 2 — the SUPERVISOR decides which capability module(s) handle this turn.

    This is the "router agent" pattern. It does NOT answer; it only classifies
    the question to one or more modules (reports / easm / aci / ...) using their
    manifest routing hints. With one module it behaves like a single agent; with
    many it can fan out (see dispatch_node). See supervisor.py for the scoring.
    """
    rr = await ctx.supervisor.route(state.get("safe_question", ""), ctx.sc)
    await ctx.fire("route", modules=rr.modules, mode=rr.mode, fallback=rr.fallback)
    return {"route_modules": rr.modules,
            "route_debug": {"scores": rr.scores, "mode": rr.mode, "fallback": rr.fallback}}


async def dispatch_node(state: ChatState, ctx: AgentContext) -> dict:
    """NODE 3 — gather evidence by running specialists IN PARALLEL.

    This is the hierarchical "supervisor -> specialists" design and the most
    important scaling idea in the system:

      * one SPECIALIST is built per routed module;
      * each specialist is scoped to ONLY its module's tools + retrievers, so
        the tool schemas of different modules never co-locate in one LLM context
        (that co-location is exactly what would break at hundreds of tools);
      * they all run concurrently via ``asyncio.gather``;
      * their findings (Chunks) are merged, ranked by relevance to the question,
        and capped into ONE numbered context block for the single answer step.
    """
    question = state.get("safe_question", "")
    # Resolve routed module ids -> live module objects, dropping any disabled ones.
    modules = [ctx.registry.module(mid) for mid in (state.get("route_modules") or [])]
    modules = [m for m in modules if m and m.enabled]
    await ctx.fire("status", stage="dispatching", specialists=[m.id for m in modules])

    # Build one specialist per module and investigate all of them at once.
    # ``return_exceptions=True`` means one specialist failing does NOT kill the
    # others — we degrade gracefully and just note the error in the trace.
    specialists = [build_specialist(m, ctx.deps, ctx.mcp) for m in modules]
    results = await asyncio.gather(
        *[sp.investigate(question, ctx.tool_ctx) for sp in specialists],
        return_exceptions=True,
    )

    chunks: list[Chunk] = []
    events: list[dict] = []
    # Merge every specialist's findings into one pool. ``strict=False`` lets zip
    # tolerate the (shouldn't-happen) length mismatch without raising.
    for module, res in zip(modules, results, strict=False):
        if isinstance(res, Exception):           # this specialist crashed -> record + skip
            events.append({"module": module.id, "ok": False, "error": str(res)})
            continue
        chunks.extend(res.chunks)
        events.extend(res.events)
        await ctx.fire("tool", module=module.id, ok=True, found=len(res.chunks))

    # Rank the merged evidence by overlap with the question, then keep the top N.
    # Ranking BEFORE capping is what prevents a relevant chunk from a late module
    # being dropped just because of insertion order.
    chunks = relevance_rank(chunks, question, MAX_CONTEXT_ENTRIES)
    # Render as "[1] text\n[2] text ..." — the [n] indices become the citation
    # markers the answer LLM uses, and that answer_node maps back to sources.
    block = "\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
    await ctx.fire("status", stage="retrieved", count=len(chunks))
    return {"context_chunks": chunks, "context_block": block, "tool_events": events}


async def answer_node(state: ChatState, ctx: AgentContext) -> dict:
    """NODE 4 — SYNTHESIZE the final answer from the gathered context.

    The single LLM "generation" step. Builds the message list (see
    build_answer_messages), runs it on the STANDARD lane (the mid-tier model),
    and — if the turn is streaming — emits each token as a ``token`` event so the
    user sees it appear live. Afterwards it parses the [n] markers the model
    wrote and maps each back to the source chunk to produce real citations.
    """
    await ctx.fire("status", stage="answering")
    messages = build_answer_messages(state, ctx)
    llm = ctx.deps.llm
    lane = Lane.STANDARD                          # answers use the standard model (see llm/lanes.py)
    if ctx.stream_tokens and ctx.emit is not None:
        # Streaming path: accumulate tokens while also emitting each one live.
        buf: list[str] = []
        async for tok in llm.stream(messages, lane=lane):
            buf.append(tok)
            await ctx.fire("token", text=tok)
        text = "".join(buf)
    else:
        # Non-streaming path (plain POST /chat): one shot, take the whole text.
        resp = await llm.complete(messages, lane=lane)
        text = resp.text

    chunks: list[Chunk] = state.get("context_chunks") or []
    from app.core.rag.citations import extract_citation_indices

    # Turn the [1], [2] markers the model emitted into structured citations,
    # mapping each index back to the chunk that occupied that slot. Out-of-range
    # markers (model hallucinated a [9] when only 8 chunks existed) are ignored.
    cited = extract_citation_indices(text)
    citations = []
    for i in cited:
        if 1 <= i <= len(chunks):
            cit = chunks[i - 1].to_citation()
            citations.append({"n": i, **cit.model_dump()})
    return {"answer": text, "citations": citations, "lane": lane.value}


async def output_guardrail_node(state: ChatState, ctx: AgentContext) -> dict:
    """NODE 5 — final safety check on the generated answer BEFORE it leaves.

    Verifies the answer is actually grounded in the retrieved chunks (catches
    hallucination), that its citations are valid, and redacts any leaked PII. If
    it blocks, we drop the citations so nothing unverified is shown as a source.
    """
    await ctx.fire("status", stage="output_guardrail")
    res = await ctx.output_guard.run(state.get("answer", ""), state.get("context_chunks") or [], ctx.sc)
    updates: dict[str, Any] = {"answer": res.text, "output_flags": res.flags}
    if res.blocked:
        updates["citations"] = []
    return updates


# --------------------------------------------------------------------------- #
# graph assembly (internal engine)
# --------------------------------------------------------------------------- #
def build_report_graph(ctx: AgentContext):
    """Wire the five nodes above into a runnable graph using our built-in engine.

    Notice how each node is added with ``lambda s: the_node(s, ctx)`` — that
    closure is how the per-request services/identity (``ctx``) get INTO a node
    while the engine still only ever calls ``node(state)``. The single
    conditional edge after the input guardrail is the turn's only branch: blocked
    -> END, otherwise -> route. Everything else is a straight line.
    """
    g = StateGraph()
    g.add_node(N_INPUT_GUARD, lambda s: input_guardrail_node(s, ctx))
    g.add_node(N_ROUTE, lambda s: route_node(s, ctx))
    g.add_node(N_GATHER, lambda s: dispatch_node(s, ctx))
    g.add_node(N_ANSWER, lambda s: answer_node(s, ctx))
    g.add_node(N_OUTPUT_GUARD, lambda s: output_guardrail_node(s, ctx))

    g.set_entry(N_INPUT_GUARD)
    # The only fork in the road: if the input guardrail blocked, skip everything
    # and go straight to END; otherwise proceed to routing.
    g.add_conditional_edges(
        N_INPUT_GUARD,
        lambda s: "blocked" if s.get("blocked") else "ok",
        {"blocked": END, "ok": N_ROUTE},
    )
    g.add_edge(N_ROUTE, N_GATHER)
    g.add_edge(N_GATHER, N_ANSWER)
    g.add_edge(N_ANSWER, N_OUTPUT_GUARD)
    g.add_edge(N_OUTPUT_GUARD, END)
    return g.compile()


# The same five nodes as a plain list. The LangGraph engine (engines.py) iterates
# this to register the IDENTICAL node set onto real LangGraph — single source of
# truth, so the two engines can never drift apart.
NODE_SPECS = [
    (N_INPUT_GUARD, input_guardrail_node),
    (N_ROUTE, route_node),
    (N_GATHER, dispatch_node),
    (N_ANSWER, answer_node),
    (N_OUTPUT_GUARD, output_guardrail_node),
]


# --------------------------------------------------------------------------- #
# planner-mode nodes (orchestrator_mode=planner)
#
# The planner graph replaces route+gather with plan + plan_dispatch and adds a
# replan_gate that can loop back to plan when evidence is missing. answer and the
# two guardrails are SHARED with the heuristic graph — only the middle changes.
# --------------------------------------------------------------------------- #
def _step_summary(res, domain: str) -> str:
    """A one-line gist of a step's findings, injected into any dependent step's
    sub-question (this is how a later step 'sees' an earlier step's result)."""
    if getattr(res, "summary", ""):
        return res.summary
    if res.chunks:
        return f"[{domain}] {res.chunks[0].text[:200]}"
    return f"[{domain}] no findings"


async def plan_node(state: ChatState, ctx: AgentContext) -> dict:
    """PLAN — the LLM brain decomposes the question into steps (see planner.py).

    Builds a fresh Planner per turn (it is stateless), asks it for a Plan, and
    publishes the steps + the set of domains involved. ``route_modules`` is set
    from the plan's domains so the shared answer node still picks a module persona
    and the TurnResult still reports what was consulted. On a replan loop,
    ``replan_notes`` from the gate is fed back in so the brain can revise."""
    from app.core.agent.planner import Planner

    planner = Planner(ctx.registry, ctx.deps.llm, ctx.settings)
    plan = await planner.plan(
        state.get("safe_question", ""), ctx.sc,
        replan_notes=state.get("replan_notes", ""),
        history=state.get("history"), summary=state.get("summary", ""),
    )
    domains = list(dict.fromkeys(s.domain for s in plan.steps))
    await ctx.fire(
        "plan",
        steps=[{"id": s.id, "domain": s.domain, "subq": s.subq, "depends_on": s.depends_on} for s in plan.steps],
        mode=plan.mode,
    )
    return {
        "plan": [s.model_dump() for s in plan.steps],
        "route_modules": domains,
        "plan_debug": {"mode": plan.mode, "synthesis": plan.synthesis,
                       "replan_round": state.get("replan_count", 0)},
    }


async def plan_dispatch_node(state: ChatState, ctx: AgentContext) -> dict:
    """DISPATCH (with dependencies) — execute the plan in dependency WAVES.

    Each wave runs every step whose dependencies are already satisfied, IN
    PARALLEL (asyncio.gather). A dependent step waits for its upstream steps and
    receives their gists prepended to its sub-question — that is the cross-module
    chain ("find the exposure, THEN find who weaponizes it"). Independent steps
    just fan out. The EXECUTOR of one step is the ordinary per-module specialist
    (build_specialist + investigate), so tool isolation, retrieval, RBAC and the
    action gate all apply exactly as in heuristic mode. Findings from all steps
    are merged, relevance-ranked, and capped into one numbered context block."""
    question = state.get("safe_question", "")
    steps: list[dict] = list(state.get("plan") or [])
    await ctx.fire("status", stage="plan_dispatch", steps=len(steps))

    summaries: dict[str, str] = {}        # step id -> gist (fed to dependents)
    done: set[str] = set()
    # Seed with any evidence already gathered on a PRIOR round, so a reflection
    # replan ADDS to the context rather than discarding what we already found
    # (relevance_rank dedupes the overlap).
    all_chunks: list[Chunk] = list(state.get("context_chunks") or [])
    events: list[dict] = []
    plan_results: list[dict] = []

    async def run_step(s: dict):
        module = ctx.registry.module(s["domain"])
        if not module or not module.enabled:
            return s, None
        subq = s["subq"]
        deps = [d for d in s.get("depends_on", []) if d in summaries]
        if deps:                          # inject upstream findings into this sub-question
            ctxt = "\n".join(f"- {summaries[d]}" for d in deps)
            subq = f"{s['subq']}\n\nUse these findings from earlier steps:\n{ctxt}"
        specialist = build_specialist(module, ctx.deps, ctx.mcp)
        res = await specialist.investigate(subq, ctx.tool_ctx)
        return s, res

    remaining = list(steps)
    guard = 0
    # The loop processes one wave per iteration. ``guard`` bounds it to the number
    # of steps (+1) — with acyclic deps that is always enough to finish.
    while remaining and guard <= len(steps) + 1:
        guard += 1
        ready = [s for s in remaining if all(d in done for d in s.get("depends_on", []))]
        if not ready:                     # unsatisfiable deps -> run the rest anyway (degrade, never hang)
            ready = remaining
        wave = await asyncio.gather(*[run_step(s) for s in ready], return_exceptions=True)
        for item in wave:
            if isinstance(item, Exception):
                continue
            s, res = item
            done.add(s["id"])
            if res is None:               # disabled/missing module
                plan_results.append({"id": s["id"], "domain": s["domain"], "subq": s["subq"], "ok": False, "found": 0})
                events.append({"step": s["id"], "domain": s["domain"], "ok": False})
                continue
            summaries[s["id"]] = _step_summary(res, s["domain"])
            all_chunks.extend(res.chunks)
            events.extend(res.events)
            plan_results.append({"id": s["id"], "domain": s["domain"], "subq": s["subq"],
                                 "ok": True, "found": len(res.chunks)})
            await ctx.fire("tool", step=s["id"], module=s["domain"], ok=True, found=len(res.chunks))
        remaining = [s for s in remaining if s["id"] not in done]

    ranked = relevance_rank(all_chunks, question, MAX_CONTEXT_ENTRIES)
    block = "\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(ranked))
    await ctx.fire("status", stage="retrieved", count=len(ranked))
    return {"context_chunks": ranked, "context_block": block,
            "tool_events": events, "plan_results": plan_results}


async def reflect_gate_node(state: ChatState, ctx: AgentContext) -> dict:
    """REFLECT GATE — runs AFTER the answer. This is what makes the loop agentic
    rather than a one-shot pipeline: a critic LLM judges whether the ANSWER
    actually addresses the question and every sub-question. If it finds a real gap
    (or a step returned no evidence) and budget remains, it loops back to PLAN with
    a note on what's missing; the next round re-plans for the gap and ACCUMULATES
    evidence (plan_dispatch seeds from the prior context).

    Conservative + BOUNDED: only an LLM-mode run reflects, only up to
    ``max_replans`` times. The deterministic path always finishes in one pass, so
    zero-infra runs and tests never loop. The decision is read by the conditional
    edge (replan -> plan, finish -> output_guard)."""
    count = state.get("replan_count", 0)
    max_replans = getattr(ctx.settings, "max_replans", 1)
    is_llm = ctx.settings.router_mode == "llm" and getattr(ctx.deps.llm, "provider", "") != "deterministic"
    if not is_llm or count >= max_replans:
        return {"needs_replan": False}

    results = state.get("plan_results") or []
    gaps = [r for r in results if not r.get("ok") or r.get("found", 0) == 0]
    question = state.get("safe_question", "")
    answer = state.get("answer", "")
    subqs = "; ".join(r.get("subq", "") for r in results) or question

    # A cheap completeness critic on the FAST lane. Reflection must never break a
    # turn, so any failure just finishes with the answer we have.
    critic = [
        ChatMessage(role="system", content=(
            "You are a strict completeness critic. Decide if the ANSWER fully addresses the "
            "QUESTION and every SUB-QUESTION. If complete, reply exactly 'COMPLETE'. Otherwise "
            "reply 'GAP: <what specifically is missing or unanswered>'.")),
        ChatMessage(role="user", content=f"QUESTION: {question}\nSUB-QUESTIONS: {subqs}\nANSWER: {answer}"),
    ]
    try:
        resp = await ctx.deps.llm.complete(critic, lane=Lane.FAST)
        verdict = (resp.text or "").strip()
    except Exception:  # noqa: BLE001 - reflection is best-effort
        return {"needs_replan": False}

    if not (verdict.upper().startswith("GAP") or gaps):
        return {"needs_replan": False}
    note = verdict if verdict.upper().startswith("GAP") else (
        "Unresolved: " + "; ".join(f"{r['domain']} ({r['subq']})" for r in gaps))
    await ctx.fire("status", stage="reflecting", round=count + 1)
    return {"needs_replan": True, "replan_count": count + 1, "replan_notes": note}


def build_planner_graph(ctx: AgentContext):
    """Wire the planner-mode graph for the built-in engine. Same shape the
    LangGraph engine mirrors (engines.py). Note the cycle: replan_gate can route
    back to plan, which the engine's max-steps fuse keeps bounded."""
    g = StateGraph()
    g.add_node(N_INPUT_GUARD, lambda s: input_guardrail_node(s, ctx))
    g.add_node(N_PLAN, lambda s: plan_node(s, ctx))
    g.add_node(N_PLAN_DISPATCH, lambda s: plan_dispatch_node(s, ctx))
    g.add_node(N_ANSWER, lambda s: answer_node(s, ctx))
    g.add_node(N_REPLAN_GATE, lambda s: reflect_gate_node(s, ctx))
    g.add_node(N_OUTPUT_GUARD, lambda s: output_guardrail_node(s, ctx))

    g.set_entry(N_INPUT_GUARD)
    g.add_conditional_edges(
        N_INPUT_GUARD,
        lambda s: "blocked" if s.get("blocked") else "ok",
        {"blocked": END, "ok": N_PLAN},
    )
    g.add_edge(N_PLAN, N_PLAN_DISPATCH)
    g.add_edge(N_PLAN_DISPATCH, N_ANSWER)
    g.add_edge(N_ANSWER, N_REPLAN_GATE)
    # The branch that makes it agentic: reflect on the answer; if incomplete, loop
    # back to PLAN (accumulating evidence); otherwise finish to the output guard.
    g.add_conditional_edges(
        N_REPLAN_GATE,
        lambda s: "replan" if s.get("needs_replan") else "finish",
        {"replan": N_PLAN, "finish": N_OUTPUT_GUARD},
    )
    g.add_edge(N_OUTPUT_GUARD, END)
    return g.compile()


# Planner node set for the LangGraph engine (mirrors build_planner_graph). Note
# the reflect gate sits AFTER answer.
PLANNER_NODE_SPECS = [
    (N_INPUT_GUARD, input_guardrail_node),
    (N_PLAN, plan_node),
    (N_PLAN_DISPATCH, plan_dispatch_node),
    (N_ANSWER, answer_node),
    (N_REPLAN_GATE, reflect_gate_node),
    (N_OUTPUT_GUARD, output_guardrail_node),
]


__all__ = [
    "build_report_graph",
    "build_planner_graph",
    "build_answer_messages",
    "NODE_SPECS",
    "PLANNER_NODE_SPECS",
    "input_guardrail_node",
    "route_node",
    "dispatch_node",
    "answer_node",
    "output_guardrail_node",
    "plan_node",
    "plan_dispatch_node",
    "reflect_gate_node",
]
