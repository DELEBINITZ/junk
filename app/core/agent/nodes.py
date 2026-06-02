"""The agent graph nodes (engine-agnostic).

Flow: input_guardrail -> route -> dispatch -> synthesize(answer) -> output_guardrail.

``route`` (the supervisor) picks the relevant module(s) from manifest hints.
``dispatch`` runs ONE specialist per routed module **in parallel**, each scoped
to its own module's tools (so tool schemas never co-locate — this is what lets
the platform scale to many modules / hundreds of tools). ``answer`` is the single
synthesize step that joins the specialists' findings and cites across pillars.

Each node takes the shared state dict + the bound :class:`AgentContext` and
returns a partial update; the same functions back both engines.
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

MAX_CONTEXT_ENTRIES = 8
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
    persona = BASE_SYSTEM
    routed = state.get("route_modules") or []
    # prepend the first routed module's persona for tone/domain
    for mid in routed:
        mod = ctx.registry.module(mid)
        if mod and mod.prompt_text:
            persona = f"{BASE_SYSTEM}\n\n{mod.prompt_text}"
            break

    messages = [ChatMessage(role="system", content=persona)]
    if state.get("summary"):
        messages.append(ChatMessage(role="system", content=f"Conversation summary so far: {state['summary']}"))
    for turn in (state.get("history") or [])[-6:]:
        role = turn.get("role", "user")
        if role in ("user", "assistant"):
            messages.append(ChatMessage(role=role, content=turn.get("content", "")))

    context_block = state.get("context_block") or "(no sources retrieved)"
    # Context goes in its own system message; the user message is the bare
    # question — keeps instruction/data separation clean and prevents the answer
    # model from treating retrieved text as part of the question.
    messages.append(ChatMessage(
        role="system",
        content=f"{CONTEXT_MARKER} (cite as [n]; if it lacks the answer, say so):\n{context_block}",
    ))
    messages.append(ChatMessage(role="user", content=state.get("safe_question") or state.get("question", "")))
    return messages


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
async def input_guardrail_node(state: ChatState, ctx: AgentContext) -> dict:
    await ctx.fire("status", stage="input_guardrail")
    res = await ctx.input_guard.run(state.get("question", ""), ctx.sc)
    if res.blocked:
        await ctx.fire("status", stage="blocked", reason=(res.reasons or ["blocked"])[0])
        return {"blocked": True, "block_reason": (res.reasons or ["blocked"])[0],
                "answer": res.text, "safe_question": state.get("question", "")}
    return {"blocked": False, "safe_question": res.text}


async def route_node(state: ChatState, ctx: AgentContext) -> dict:
    rr = await ctx.supervisor.route(state.get("safe_question", ""), ctx.sc)
    await ctx.fire("route", modules=rr.modules, mode=rr.mode, fallback=rr.fallback)
    return {"route_modules": rr.modules,
            "route_debug": {"scores": rr.scores, "mode": rr.mode, "fallback": rr.fallback}}


async def dispatch_node(state: ChatState, ctx: AgentContext) -> dict:
    """Dispatch one specialist per routed module IN PARALLEL. Each specialist is
    scoped to its own module's tools (no cross-module tool-schema bloat). Their
    findings are merged, ranked by relevance, and capped into one org-scoped
    context block for the single synthesize step."""
    question = state.get("safe_question", "")
    modules = [ctx.registry.module(mid) for mid in (state.get("route_modules") or [])]
    modules = [m for m in modules if m and m.enabled]
    await ctx.fire("status", stage="dispatching", specialists=[m.id for m in modules])

    specialists = [build_specialist(m, ctx.deps, ctx.mcp) for m in modules]
    results = await asyncio.gather(
        *[sp.investigate(question, ctx.tool_ctx) for sp in specialists],
        return_exceptions=True,
    )

    chunks: list[Chunk] = []
    events: list[dict] = []
    for module, res in zip(modules, results, strict=False):
        if isinstance(res, Exception):
            events.append({"module": module.id, "ok": False, "error": str(res)})
            continue
        chunks.extend(res.chunks)
        events.extend(res.events)
        await ctx.fire("tool", module=module.id, ok=True, found=len(res.chunks))

    chunks = relevance_rank(chunks, question, MAX_CONTEXT_ENTRIES)
    block = "\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
    await ctx.fire("status", stage="retrieved", count=len(chunks))
    return {"context_chunks": chunks, "context_block": block, "tool_events": events}


async def answer_node(state: ChatState, ctx: AgentContext) -> dict:
    await ctx.fire("status", stage="answering")
    messages = build_answer_messages(state, ctx)
    llm = ctx.deps.llm
    lane = Lane.STANDARD
    if ctx.stream_tokens and ctx.emit is not None:
        buf: list[str] = []
        async for tok in llm.stream(messages, lane=lane):
            buf.append(tok)
            await ctx.fire("token", text=tok)
        text = "".join(buf)
    else:
        resp = await llm.complete(messages, lane=lane)
        text = resp.text

    chunks: list[Chunk] = state.get("context_chunks") or []
    from app.core.rag.citations import extract_citation_indices

    cited = extract_citation_indices(text)
    citations = []
    for i in cited:
        if 1 <= i <= len(chunks):
            cit = chunks[i - 1].to_citation()
            citations.append({"n": i, **cit.model_dump()})
    return {"answer": text, "citations": citations, "lane": lane.value}


async def output_guardrail_node(state: ChatState, ctx: AgentContext) -> dict:
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
    g = StateGraph()
    g.add_node(N_INPUT_GUARD, lambda s: input_guardrail_node(s, ctx))
    g.add_node(N_ROUTE, lambda s: route_node(s, ctx))
    g.add_node(N_GATHER, lambda s: dispatch_node(s, ctx))
    g.add_node(N_ANSWER, lambda s: answer_node(s, ctx))
    g.add_node(N_OUTPUT_GUARD, lambda s: output_guardrail_node(s, ctx))

    g.set_entry(N_INPUT_GUARD)
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


# node specs reused by the LangGraph engine
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
