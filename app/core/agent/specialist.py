"""Per-module specialist sub-agent.

The supervisor dispatches ONE specialist per routed module, in parallel. Each
specialist is **scoped to its own module's tools** — it retrieves from that
module's corpus and calls only that module's tools, then returns findings. This
is the load-bearing scaling property: tool schemas never co-locate, so the system
holds many modules / hundreds of tools without bloating any one context.

Two planners:
  * heuristic (default, infra-free) — auto-invoke the module's read tools whose
    args are satisfiable from the question + run its retrievers.
  * LLM tool-calling (``ROUTER_MODE=llm`` + a real provider) — the specialist's
    LLM picks among ONLY this module's tools via function-calling, bounded by
    ``max_tool_iterations``.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.contracts import Chunk, SpecialistResult, ToolContext, ToolResult
from app.core.llm.base import ChatMessage, Lane, LLMToolSpec

PER_SPECIALIST_MAX = 6
_WORD = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# shared helpers (also used by the dispatch node)
# --------------------------------------------------------------------------- #
def relevance_rank(chunks: list[Chunk], question: str, cap: int) -> list[Chunk]:
    """Rank by lexical overlap with the question, tie-broken by retrieval score."""
    qtok = set(_WORD.findall(question.lower()))

    def key(c: Chunk) -> tuple[int, float]:
        return (len(qtok & set(_WORD.findall(c.text.lower()))), c.score)

    return sorted([c for c in chunks if c.text.strip()], key=key, reverse=True)[:cap]


def autocall_args(tool, question: str):
    """Args to auto-invoke a read tool, or None if its required args can't be
    satisfied from the question (then it's a targeted tool, not a gatherer)."""
    fields = getattr(tool.args_schema, "model_fields", {})
    args: dict[str, Any] = {}
    for fname in ("query", "q", "question", "text"):
        if fname in fields:
            args[fname] = question
            break
    for fname, field in fields.items():
        if fname in args:
            continue
        required = field.is_required() if hasattr(field, "is_required") else field.default is None
        if required:
            return None
    return args


def summarize_tool_result(name: str, result: ToolResult) -> str:
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
    def __init__(self, module, deps, mcp) -> None:
        self.module = module
        self.deps = deps
        self.mcp = mcp
        self.id = module.id

    def _read_tools(self) -> list:
        return [t for t in self.module.tools.values() if not t.side_effecting]

    def _use_llm_planner(self) -> bool:
        s = self.deps.settings
        return (
            s.router_mode == "llm"
            and getattr(self.deps.llm, "provider", "") != "deterministic"
            and bool(self._read_tools())
        )

    def _fold(self, name: str, out, chunks: list[Chunk], events: list[dict], org_id: str) -> None:
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
        for r in self.module.retrievers.values():
            try:
                chunks.extend(await r.retrieve(question, {}, ctx))
            except Exception as exc:  # noqa: BLE001
                events.append({"tool": r.id, "ok": False, "error": str(exc), "module": self.id})

    async def _heuristic_tools(self, question: str, ctx: ToolContext, chunks: list[Chunk], events: list[dict]) -> None:
        for tname, tool in self.module.tools.items():
            if tool.side_effecting:
                continue
            args = autocall_args(tool, question)
            if args is None:
                continue
            out = await self.mcp.call_tool(tname, args, ctx)
            self._fold(tname, out, chunks, events, ctx.org_id)

    async def _llm_tools(self, question: str, ctx: ToolContext, chunks: list[Chunk], events: list[dict]) -> None:
        tools = [LLMToolSpec(name=t.name, description=t.description,
                             parameters=t.args_schema.model_json_schema()) for t in self._read_tools()]
        persona = self.module.prompt_text or f"You are the {self.module.manifest.display_name} specialist."
        messages = [ChatMessage(role="system", content=persona),
                    ChatMessage(role="user", content=question)]
        for _ in range(max(1, self.deps.settings.max_tool_iterations)):
            resp = await self.deps.llm.complete_with_tools(messages, tools, lane=Lane.FAST, tool_choice="auto")
            if not resp.tool_calls:
                break
            observations = []
            for tc in resp.tool_calls:
                out = await self.mcp.call_tool(tc.name, tc.arguments, ctx)
                self._fold(tc.name, out, chunks, events, ctx.org_id)
                observations.append(f"{tc.name} -> {summarize_tool_result(tc.name, out) if isinstance(out, ToolResult) else out.code}")
            messages.append(ChatMessage(role="system", content="Tool observations: " + "; ".join(observations)))

    async def investigate(self, question: str, ctx: ToolContext) -> SpecialistResult:
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
    """Use a module's custom specialist (manifest.specialist) if it ships one,
    else the generic specialist. Custom factory signature: ``(module, deps, mcp)``."""
    factory = module.manifest.specialist
    if factory is not None:
        return factory(module, deps, mcp)
    return GenericSpecialist(module, deps, mcp)


__all__ = ["GenericSpecialist", "build_specialist", "relevance_rank",
           "autocall_args", "summarize_tool_result", "PER_SPECIALIST_MAX"]
