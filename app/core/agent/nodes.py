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
    # Only the last 6 turns — recent context matters, older context is covered by
    # the rolling summary above. This bounds prompt size on long conversations.
    for turn in (state.get("history") or [])[-6:]:
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


__all__ = [
    "build_report_graph",
    "build_answer_messages",
    "NODE_SPECS",
    "input_guardrail_node",
    "route_node",
    "dispatch_node",
    "answer_node",
    "output_guardrail_node",
]
