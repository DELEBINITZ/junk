"""Per-module SPECIALIST sub-agent — the unit the supervisor fans out to.

THE BIG IDEA (why this file exists):
A naive agent puts EVERY tool from EVERY module into one giant prompt. With 4
tools that is fine; with 400 tools the model drowns — the context is bloated,
routing gets worse, and cost explodes. The fix is hierarchy:

    supervisor (route_node)  -> picks the relevant module(s)
        |
        +-- specialist(reports)   sees ONLY reports' tools + corpus
        +-- specialist(easm)      sees ONLY easm's tools
        +-- specialist(aci)       sees ONLY aci's tools
        (these run in parallel; their findings are merged by the synthesize step)

Each specialist is "tool-isolated": it never sees another module's tools, so
schemas never co-locate. That is what lets the platform grow to many modules and
hundreds of tools without any single agent's context blowing up.

A specialist "investigates" — it RETRIEVES from its module's corpus and CALLS
its module's read-only tools — then returns Chunks (findings). It does NOT write
the final answer; the single synthesize step (answer_node) does that across all
specialists. There are two ways a specialist decides which tools to call:

  * heuristic planner (default, needs no LLM/GPU) — auto-invoke each read tool
    whose required args can be filled straight from the question, and run the
    module's retrievers. Deterministic and infra-free.
  * LLM tool-calling planner (``ROUTER_MODE=llm`` + a real model) — let the
    specialist's own LLM pick among ONLY this module's tools via function
    calling, looping up to ``max_tool_iterations`` times.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from app.core.contracts import Chunk, SpecialistResult, ToolContext, ToolResult
from app.core.llm.base import ChatMessage, Lane, LLMToolSpec

# Per-specialist evidence cap. Each specialist returns at most this many chunks,
# so a single chatty module can't dominate the merged context downstream.
PER_SPECIALIST_MAX = 6
_WORD = re.compile(r"[a-z0-9]+")     # crude tokenizer used by relevance ranking

# Cache of tool-description embedding vectors, so we embed a module's tool surface
# ONCE (keyed by module id + embedder provider + tool names) instead of every turn
# — important when an integrated MCP server exposes a large, stable tool set.
_TOOL_VEC_CACHE: dict[tuple, list] = {}


def _cosine(a, b) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    return float(va @ vb / (na * nb)) if na and nb else 0.0


def _lexical_select(tools: list, question: str, cap: int) -> list:
    """Fallback tool ranker (keyword overlap) when no embedder is available."""
    qtok = set(_WORD.findall(question.lower()))

    def score(t) -> int:
        return len(qtok & set(_WORD.findall(f"{t.name} {t.description}".lower())))

    return sorted(tools, key=score, reverse=True)[:cap]


# --------------------------------------------------------------------------- #
# shared helpers (also used by the dispatch node)
# --------------------------------------------------------------------------- #
def relevance_rank(chunks: list[Chunk], question: str, cap: int) -> list[Chunk]:
    """Rank chunks by how many question words they contain, then keep the top
    ``cap``. Ties are broken by the chunk's own retrieval ``score``.

    This is a cheap, deterministic lexical ranker (no model needed). The real
    quality win — a cross-encoder reranker — lives in rag/reranker.py; this one
    just makes the merge step sensible with zero infra.
    """
    qtok = set(_WORD.findall(question.lower()))

    def key(c: Chunk) -> tuple[int, float]:
        # (overlap count, retrieval score) — sorted descending below.
        return (len(qtok & set(_WORD.findall(c.text.lower()))), c.score)

    # Drop empty chunks, sort best-first, then DEDUPE by exact text so duplicate
    # passages (same doc retrieved by two tools/steps, or across replan rounds)
    # don't fill the context and crowd out distinct evidence. Dedupe after sorting
    # keeps the best-ranked copy.
    ranked = sorted([c for c in chunks if c.text.strip()], key=key, reverse=True)
    out: list[Chunk] = []
    seen: set[str] = set()
    for c in ranked:
        sig = c.text.strip().lower()
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
        if len(out) >= cap:
            break
    return out


async def rank_evidence(chunks: list[Chunk], question: str, cap: int, embedder=None) -> list[Chunk]:
    """SEMANTIC merge-ranker for the final answer context — the highest-leverage
    ranking in the turn (it decides which evidence the answer LLM actually sees).

    Ranks the merged cross-specialist + tool evidence by EMBEDDING similarity to the
    question (by meaning, robust to wording) instead of crude keyword overlap, then
    dedupes by exact text and keeps the top ``cap``. The per-collection cross-encoder
    reranker only runs INSIDE retrieval; this is the equivalent quality step for the
    POOLED evidence (RAG chunks + tool outputs from every module/plan step).

    Falls back to the lexical :func:`relevance_rank` when no embedder is wired or on
    any failure — ranking must never break a turn.
    """
    clean = [c for c in chunks if c.text.strip()]
    if embedder is None or len(clean) <= 1:
        return relevance_rank(chunks, question, cap)
    try:
        qv = await embedder.embed_query(question)
        vecs = await embedder.embed([c.text for c in clean])
        order = sorted(range(len(clean)), key=lambda i: _cosine(qv, vecs[i]), reverse=True)
        out: list[Chunk] = []
        seen: set[str] = set()
        for i in order:
            c = clean[i]
            sig = c.text.strip().lower()
            if sig in seen:
                continue
            seen.add(sig)
            out.append(c)
            if len(out) >= cap:
                break
        return out
    except Exception:  # noqa: BLE001 - ranking must never break a turn
        return relevance_rank(chunks, question, cap)


def autocall_args(tool, question: str):
    """For the HEURISTIC planner: figure out the args to auto-invoke a read tool,
    or return ``None`` meaning "this tool needs specific args I can't infer from
    a plain question — skip it" (it's a targeted tool, not a general gatherer).

    Logic: if the tool has a free-text field (query/q/question/text), pass the
    whole question into it. Then if ANY other required field is still unfilled,
    bail (None) — we won't guess structured args heuristically.
    """
    fields = getattr(tool.args_schema, "model_fields", {})
    args: dict[str, Any] = {}
    for fname in ("query", "q", "question", "text"):
        if fname in fields:
            args[fname] = question
            break
    for fname, field in fields.items():
        if fname in args:
            continue
        # pydantic v2: a field is required if it has no default.
        required = field.is_required() if hasattr(field, "is_required") else field.default is None
        if required:
            return None
    return args


def summarize_tool_result(name: str, result: ToolResult) -> str:
    """One-line gist of a tool result, used as a fallback "chunk" when the tool
    returned structured ``data`` rather than citations — so even non-document
    tools contribute readable evidence to the context."""
    if result.citations:
        return result.citations[0].snippet or result.citations[0].title
    if not result.data:
        return f"{name}: ok"
    bits = []
    for k, v in list(result.data.items())[:6]:
        bits.append(f"{k}={len(v)} item(s)" if isinstance(v, (list, dict)) else f"{k}={v}")
    return f"{name}: " + ", ".join(bits)


# --------------------------------------------------------------------------- #
# generic specialist
# --------------------------------------------------------------------------- #
class GenericSpecialist:
    """The default specialist used by any module that doesn't ship a custom one.

    It implements the ``Specialist`` protocol from contracts.py: a single
    ``investigate(question, ctx)`` method that returns a ``SpecialistResult``.
    A module can override this with its own class via ``manifest.specialist``
    (see build_specialist) — e.g. for a multi-hop or domain-specific strategy.
    """

    def __init__(self, module, deps, mcp) -> None:
        self.module = module     # the one capability module this specialist owns
        self.deps = deps         # CoreDeps (llm, settings, ...)
        self.mcp = mcp           # the MCP client — ALL tool calls go through it (RBAC+gate)
        self.id = module.id

    def _read_tools(self) -> list:
        """ONLY this module's non-side-effecting tools. Two limits at once:
        (1) tool isolation — never another module's tools; (2) read-only — a
        specialist never triggers a side-effecting action on its own; those go
        through the human approval gate. Both are core safety properties."""
        return [t for t in self.module.tools.values() if not t.side_effecting]

    async def _select_tools(self, question: str, cap: int) -> list:
        """PRODUCTION TOOL SELECTION (semantic tool-RAG) — the context-flood guard.

        Returns the ``cap`` tools most relevant to the question, so the agent
        CHOOSES correctly even when a module (e.g. an integrated MCP server)
        exposes hundreds of tools, without flooding the LLM context with every
        schema. Ranks by EMBEDDING similarity between the question and each tool's
        name+description — robust to wording, unlike keyword overlap — and caches
        the tool vectors per module so the surface is embedded once, not per turn.
        Falls back to the lexical ranker if no embedder is wired, and never raises
        (a selection failure must not break a turn)."""
        tools = self._read_tools()
        if len(tools) <= cap:
            return tools
        embedder = getattr(getattr(self.deps, "rag", None), "embedder", None)
        if embedder is None:
            return _lexical_select(tools, question, cap)
        try:
            key = (self.id, getattr(embedder, "provider", "?"), tuple(t.name for t in tools))
            vecs = _TOOL_VEC_CACHE.get(key)
            if vecs is None:
                vecs = await embedder.embed([f"{t.name}: {t.description}" for t in tools])
                _TOOL_VEC_CACHE[key] = vecs
            qv = await embedder.embed_query(question)
            order = sorted(range(len(tools)), key=lambda i: _cosine(qv, vecs[i]), reverse=True)
            return [tools[i] for i in order[:cap]]
        except Exception:  # noqa: BLE001 - selection must never break a turn
            return _lexical_select(tools, question, cap)

    async def _rank_candidates(self, cands: list[dict], question: str, cap: int) -> list[dict]:
        """Semantic shortlist over UNIFIED candidates (local + discovered), each a
        dict with name/description. Same embedding tool-RAG as ``_select_tools`` but
        over dicts, so it ranks local Tool stubs and dynamically-discovered remote
        tools together, then keeps the top ``cap``."""
        if len(cands) <= cap:
            return cands
        embedder = getattr(getattr(self.deps, "rag", None), "embedder", None)
        if embedder is None:
            qtok = set(_WORD.findall(question.lower()))

            def sc(c) -> int:
                return len(qtok & set(_WORD.findall(f"{c['name']} {c.get('description', '')}".lower())))

            return sorted(cands, key=sc, reverse=True)[:cap]
        try:
            qv = await embedder.embed_query(question)
            vecs = await embedder.embed([f"{c['name']}: {c.get('description', '')}" for c in cands])
            order = sorted(range(len(cands)), key=lambda i: _cosine(qv, vecs[i]), reverse=True)
            return [cands[i] for i in order[:cap]]
        except Exception:  # noqa: BLE001
            return cands[:cap]

    async def _gather_candidates(self, question: str, ctx: ToolContext, cap: int) -> list[dict]:
        """Build the LLM's tool candidate set: this module's LOCAL read tools PLUS
        any tools dynamically DISCOVERED from its remote MCP server (pure dynamic),
        de-duped (a local declaration wins over a discovered tool of the same name),
        then semantically shortlisted to ``cap``. This is what lets the agent choose
        correctly among a remote server's LIVE toolset with no local declaration,
        while never flooding the context with every schema."""
        cands: list[dict] = [
            {"name": t.name, "description": t.description, "parameters": t.args_schema.model_json_schema()}
            for t in self._read_tools()
        ]
        discover = getattr(self.mcp, "discover_tools", None)
        if discover is not None:
            try:
                for d in await discover(self.id, ctx):
                    cands.append({"name": d["name"], "description": d.get("description", ""),
                                  "parameters": d.get("parameters", {})})
            except Exception:  # noqa: BLE001 - discovery is best-effort
                pass
        seen: set[str] = set()
        uniq: list[dict] = []
        for c in cands:
            if c["name"] in seen:
                continue
            seen.add(c["name"])
            uniq.append(c)
        return await self._rank_candidates(uniq, question, cap)

    def _use_llm_planner(self) -> bool:
        """Pick the planner. Use the LLM tool-calling planner only if configured
        (``router_mode=llm``), a REAL model is wired (not the deterministic stub),
        and this module actually has read tools to choose among. Otherwise fall
        back to the infra-free heuristic planner."""
        s = self.deps.settings
        return (
            s.router_mode == "llm"
            and getattr(self.deps.llm, "provider", "") != "deterministic"
            and bool(self._read_tools())
        )

    def _fold(self, name: str, out, chunks: list[Chunk], events: list[dict], org_id: str) -> None:
        """Turn one tool OUTCOME into findings. Tool calls return data, not
        exceptions (the "errors-as-data" contract), so we branch on the type:
          * ToolResult -> convert each citation into a Chunk (the evidence), and
            add a one-line summary chunk if the tool returned structured data.
          * ToolError  -> just record it in the trace; no chunk.
        Every Chunk is stamped with ``org_id`` and ``source=this module`` so the
        provenance (and tenant) of each piece of evidence is never lost."""
        if isinstance(out, ToolResult):
            events.append({"tool": name, "ok": True, "module": self.id})
            for c in out.citations:
                chunks.append(Chunk(id=f"{name}:{c.doc_id}", text=c.snippet or c.title, org_id=org_id,
                                    source=self.id, doc_id=c.doc_id, title=c.title, score=c.score,
                                    published_at=c.published_at))
            if out.data:
                chunks.append(Chunk(id=f"{name}:summary", text=summarize_tool_result(name, out),
                                    org_id=org_id, source=self.id, doc_id=name, title=name))
        else:
            events.append({"tool": name, "ok": False, "code": getattr(out, "code", "error"), "module": self.id})

    async def _retrieve(self, question: str, ctx: ToolContext, chunks: list[Chunk], events: list[dict]) -> None:
        """Run the module's RAG retrievers (if any). Document-corpus modules like
        ``reports`` bind a retriever here; pure tool modules like ``easm``/``aci``
        have none and rely entirely on their tools. A failing retriever is logged,
        not raised — the investigation continues with whatever else it finds."""
        for r in self.module.retrievers.values():
            try:
                chunks.extend(await r.retrieve(question, {}, ctx))
            except Exception as exc:  # noqa: BLE001
                events.append({"tool": r.id, "ok": False, "error": str(exc), "module": self.id})

    async def _heuristic_tools(self, question: str, ctx: ToolContext, chunks: list[Chunk], events: list[dict]) -> None:
        """HEURISTIC planner: for each read tool whose args we can fill from the
        question (see autocall_args), call it through the MCP boundary and fold
        the result into findings. No LLM involved — fully deterministic."""
        # Only consider the shortlisted read tools (flood guard) — _select_tools
        # already excludes side-effecting tools.
        for tool in await self._select_tools(question, self.deps.settings.max_tools_advertised):
            # Skip tools a module marked auto_invoke=False (LLM/planner-only, e.g.
            # a RAG search tool that would duplicate the module's bound retriever).
            if not tool.auto_invoke:
                continue
            args = autocall_args(tool, question)
            if args is None:                       # can't infer args -> skip this tool
                continue
            out = await self.mcp.call_tool(tool.name, args, ctx)
            self._fold(tool.name, out, chunks, events, ctx.org_id)

    async def _llm_tools(self, question: str, ctx: ToolContext, chunks: list[Chunk], events: list[dict]) -> None:
        """LLM tool-calling planner: advertise this module's tools to the model and
        let it choose which to call (and with what args) via function calling,
        looping until it stops asking for tools or we hit ``max_tool_iterations``.
        Candidates are this module's LOCAL read tools PLUS tools dynamically
        DISCOVERED from its remote MCP server (``_gather_candidates``), semantically
        shortlisted so a huge live toolset never floods context. The model cannot
        pick another module's tool (tool isolation)."""
        candidates = await self._gather_candidates(question, ctx, self.deps.settings.max_tools_advertised)
        tools = [LLMToolSpec(name=c["name"], description=c["description"],
                             parameters=c["parameters"]) for c in candidates]
        persona = self.module.prompt_text or f"You are the {self.module.manifest.display_name} specialist."
        messages = [ChatMessage(role="system", content=persona),
                    ChatMessage(role="user", content=question)]
        for _ in range(max(1, self.deps.settings.max_tool_iterations)):
            # FAST lane — tool planning is cheap, the heavy model is for the answer.
            resp = await self.deps.llm.complete_with_tools(messages, tools, lane=Lane.FAST, tool_choice="auto")
            if not resp.tool_calls:                # model is done asking for tools
                break
            observations = []
            for tc in resp.tool_calls:
                out = await self.mcp.call_tool(tc.name, tc.arguments, ctx)   # still goes through RBAC+gate
                self._fold(tc.name, out, chunks, events, ctx.org_id)
                observations.append(f"{tc.name} -> {summarize_tool_result(tc.name, out) if isinstance(out, ToolResult) else out.code}")
            # Feed the tool results back so the model can decide its next move
            # (classic ReAct-style observe-then-act tool loop).
            messages.append(ChatMessage(role="system", content="Tool observations: " + "; ".join(observations)))

    async def investigate(self, question: str, ctx: ToolContext) -> SpecialistResult:
        """The Specialist protocol method. Orchestrates one module's evidence
        gathering: retrieve from the corpus, then run the chosen tool planner,
        then rank + cap the findings. Returns a SpecialistResult the dispatch
        node merges with the other specialists'."""
        chunks: list[Chunk] = []
        events: list[dict] = []
        await self._retrieve(question, ctx, chunks, events)
        if self._use_llm_planner():
            await self._llm_tools(question, ctx, chunks, events)
        else:
            await self._heuristic_tools(question, ctx, chunks, events)
        ranked = relevance_rank(chunks, question, PER_SPECIALIST_MAX)
        return SpecialistResult(module_id=self.id, chunks=ranked, events=events)


def build_specialist(module, deps, mcp):
    """Factory: use a module's CUSTOM specialist if its manifest declares one
    (``manifest.specialist``), otherwise the GenericSpecialist above. This is the
    extension point — a module with special reasoning needs ships its own
    ``Specialist`` without the core knowing anything about it. Custom factory
    signature is ``(module, deps, mcp)``."""
    factory = module.manifest.specialist
    if factory is not None:
        return factory(module, deps, mcp)
    return GenericSpecialist(module, deps, mcp)


__all__ = ["GenericSpecialist", "build_specialist", "relevance_rank", "rank_evidence",
           "autocall_args", "summarize_tool_result", "PER_SPECIALIST_MAX"]
