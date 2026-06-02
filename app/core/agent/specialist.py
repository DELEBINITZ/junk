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

from app.core.contracts import Chunk, SpecialistResult, ToolContext, ToolResult
from app.core.llm.base import ChatMessage, Lane, LLMToolSpec

# Per-specialist evidence cap. Each specialist returns at most this many chunks,
# so a single chatty module can't dominate the merged context downstream.
PER_SPECIALIST_MAX = 6
_WORD = re.compile(r"[a-z0-9]+")     # crude tokenizer used by relevance ranking


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

    # Drop empty chunks, sort best-first, keep ``cap``.
    return sorted([c for c in chunks if c.text.strip()], key=key, reverse=True)[:cap]


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
        for tname, tool in self.module.tools.items():
            if tool.side_effecting:
                continue
            args = autocall_args(tool, question)
            if args is None:                       # can't infer args -> skip this tool
                continue
            out = await self.mcp.call_tool(tname, args, ctx)
            self._fold(tname, out, chunks, events, ctx.org_id)

    async def _llm_tools(self, question: str, ctx: ToolContext, chunks: list[Chunk], events: list[dict]) -> None:
        """LLM tool-calling planner: advertise ONLY this module's read tools to
        the model and let it choose which to call (and with what args) via
        function calling, looping until it stops asking for tools or we hit
        ``max_tool_iterations``. Note ``self._read_tools()`` — the model literally
        cannot pick a tool from another module, because it never sees one."""
        tools = [LLMToolSpec(name=t.name, description=t.description,
                             parameters=t.args_schema.model_json_schema()) for t in self._read_tools()]
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


__all__ = ["GenericSpecialist", "build_specialist", "relevance_rank",
           "autocall_args", "summarize_tool_result", "PER_SPECIALIST_MAX"]
