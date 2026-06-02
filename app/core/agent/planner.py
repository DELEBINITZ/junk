"""The PLANNER — the LLM "brain" of planner-mode orchestration.

WHERE THIS SITS: in planner mode the agent graph is
    input_guard -> PLAN -> dispatch -> replan_gate -> synthesize -> output_guard
and THIS file owns the PLAN step. Instead of the heuristic supervisor picking a
couple of modules by keyword overlap, the planner reads compact CAPABILITY CARDS
(one per module the caller may see) and decides a PLAN: a list of steps, each
naming a target domain + a self-contained sub-question, with optional
``depends_on`` links when one step needs an earlier step's findings first.

WHY CARDS, NOT RAW TOOLS: the planner must stay cheap and un-bloated even with
hundreds of tools across many modules. So the brain sees a one-line card per
domain (what it's good for, which tools it has) — NOT every tool schema. The
specialists (specialist.py) hold the full tool detail and execute each step. Two
tiers of reasoning: planner = strategy (which domains, what sub-questions, what
order), specialist = tactics (which tools, what args). Neither context bloats.

TWO MODES (mirrors the supervisor):
  * deterministic / heuristic (default, no GPU): reuse the supervisor's routing to
    pick domains, then emit one PARALLEL step per domain (no dependencies). This is
    why the zero-infra path and the existing tests still work in planner mode.
  * llm (``router_mode=llm`` + a real model): prompt the model with the cards and
    parse a JSON plan that MAY include cross-step ``depends_on`` dependencies.

SAFETY: the plan is always validated — unknown domains dropped, steps capped to
``max_plan_steps``, and ``depends_on`` may only reference EARLIER steps, which
makes the dependency graph acyclic by construction (the executor can't deadlock).
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from app.core.agent.supervisor import Supervisor
from app.core.llm.base import ChatMessage, Lane
from app.core.security.context import SecurityContext


class PlanStep(BaseModel):
    """One unit of the plan: ask ``domain`` to investigate ``subq``. ``depends_on``
    lists the ids of earlier steps whose findings must be gathered first and fed
    into this step's sub-question (the cross-module dependency mechanism)."""

    id: str
    domain: str                                     # a capability module id (reports, easm, aci, ...)
    subq: str                                       # the self-contained sub-question for that domain
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """A full plan: the ordered steps plus a free-text ``synthesis`` goal the final
    answer step uses to combine findings. ``mode`` records how it was produced
    (heuristic vs llm) for observability."""

    steps: list[PlanStep] = Field(default_factory=list)
    synthesis: str = ""
    mode: str = "heuristic"


# Matches a ```json ... ``` fenced block, so we can recover JSON even if the model
# wraps it in markdown (a common LLM habit).
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class Planner:
    def __init__(self, registry, llm, settings) -> None:
        self.registry = registry
        self.llm = llm
        self.settings = settings
        # The deterministic/heuristic plan reuses the supervisor's domain selection,
        # capped at planner_max_fanout domains.
        self.supervisor = Supervisor(registry, llm, settings, max_fanout=settings.planner_max_fanout)

    def _use_llm(self) -> bool:
        # Only plan with the model when LLM mode is on AND a real provider is wired
        # (the deterministic stub can't produce a meaningful JSON plan).
        return self.settings.router_mode == "llm" and getattr(self.llm, "provider", "") != "deterministic"

    def _cards(self, sc: SecurityContext) -> str:
        """Build the compact capability cards the planner reasons over — one line
        per module the caller is entitled to see (capability_view already applies
        license + RBAC filtering, so the planner can never plan against a hidden
        module). Each card: id | what it's good for (routing-hint intents) | tools."""
        view = self.registry.capability_view(sc)
        # Group routing-hint intents by module for the "good for" blurb.
        intents_by_mod: dict[str, list[str]] = {}
        for mid, hint in view.routing:
            intents_by_mod.setdefault(mid, []).extend(hint.intents)
        lines: list[str] = []
        for mid in dict.fromkeys(view.module_ids):
            m = self.registry.module(mid)
            if not m:
                continue
            # Cap the tool names shown so a module with a HUGE remote tool surface
            # doesn't bloat the planner prompt — the card stays description-led; the
            # specialist does the fine-grained tool selection later (_select_tools).
            tools = [t.name for t in m.tools.values()]
            cap = 8
            shown = ", ".join(tools[:cap]) + (f", …(+{len(tools) - cap} more)" if len(tools) > cap else "")
            good = ", ".join(dict.fromkeys(intents_by_mod.get(mid, []))) or m.manifest.description
            lines.append(
                f"- {mid}: {m.manifest.description} | good for: {good} | tools: {shown or 'RAG search'}"
            )
        return "\n".join(lines)

    async def plan(
        self, question: str, sc: SecurityContext, *,
        replan_notes: str = "", history: list | None = None, summary: str = "",
    ) -> Plan:
        """Produce a Plan. Try the LLM planner when enabled; on any failure (or in
        deterministic mode) fall back to the heuristic plan so a turn always gets
        a usable plan. ``history``/``summary`` give the planner conversational
        context so it can decompose FOLLOW-UP questions ("and which are critical?")
        correctly instead of planning the bare words in isolation."""
        if self._use_llm():
            p = await self._llm_plan(question, sc, replan_notes, history or [], summary)
            if p and p.steps:
                return p
        return await self._heuristic_plan(question, sc)

    async def _heuristic_plan(self, question: str, sc: SecurityContext) -> Plan:
        """Deterministic plan: the supervisor picks the domains, and we emit one
        independent (parallel) step per domain — equivalent to the v1 fan-out, but
        expressed as a Plan so the rest of the planner pipeline is identical."""
        rr = await self.supervisor.route(question, sc)
        steps = [PlanStep(id=f"s{i + 1}", domain=mid, subq=question) for i, mid in enumerate(rr.modules)]
        return Plan(
            steps=steps,
            synthesis="Answer the user's question from the gathered findings; cite every claim.",
            mode=f"heuristic:{rr.mode}",
        )

    async def _llm_plan(
        self, question: str, sc: SecurityContext, replan_notes: str,
        history: list | None = None, summary: str = "",
    ) -> Plan | None:
        """Ask the model for a JSON plan over the capability cards. Returns None on
        any parse/validation failure so the caller falls back to the heuristic plan
        — the planner must never hard-fail a turn."""
        view = self.registry.capability_view(sc)
        available = list(dict.fromkeys(view.module_ids))
        if not available:
            return None
        cards = self._cards(sc)
        sys = (
            "You are the planning brain of a multi-tenant security-intelligence assistant. "
            "Decompose the user's question into the MINIMAL plan over the capability domains below. "
            "Use a single step when one domain suffices; use multiple steps only for genuinely "
            "cross-domain questions. When a step needs another step's findings first, put that step's "
            "id in depends_on (a step may ONLY depend on steps listed before it). Each subq must be "
            "self-contained — resolve pronouns/follow-ups using the conversation context if given.\n\n"
            f"Domains:\n{cards}\n\n"
            'Reply with ONLY JSON of the form: '
            '{"steps":[{"id":"s1","domain":"<domain-id>","subq":"...","depends_on":[]}],"synthesis":"..."}'
        )
        # Give the planner just enough conversation context to resolve follow-ups,
        # bounded so it never bloats the planning prompt.
        convo = ""
        if summary:
            convo += f"Summary: {summary}\n"
        for turn in (history or [])[-4:]:
            role, content = turn.get("role", "user"), turn.get("content", "")
            if role in ("user", "assistant") and content:
                convo += f"{role}: {content[:200]}\n"
        user = question if not replan_notes else f"{question}\n\n(Revise the plan. {replan_notes})"
        if convo:
            user = f"[Conversation so far]\n{convo}\n[Current question]\n{user}"
        try:
            resp = await self.llm.complete(
                [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)],
                lane=Lane.FAST,            # planning is cheap reasoning -> fast lane
            )
        except Exception:
            return None
        data = self._parse(resp.text)
        if not data:
            return None
        return self._validate(data, available)

    def _parse(self, text: str) -> dict | None:
        """Best-effort extract a JSON object from the model's reply (handles raw
        JSON, ```json fences, or JSON embedded in prose)."""
        if not text:
            return None
        m = _FENCE.search(text)
        raw = m.group(1) if m else text
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None

    def _validate(self, data: dict, available: list[str]) -> Plan | None:
        """Turn raw parsed JSON into a safe Plan. Drops steps targeting unknown
        domains, caps to ``max_plan_steps``, and keeps only ``depends_on`` ids that
        refer to ALREADY-seen (earlier) steps — which guarantees an acyclic
        dependency graph the executor can always finish."""
        raw_steps = data.get("steps") or []
        steps: list[PlanStep] = []
        seen_ids: set[str] = set()
        for i, rs in enumerate(raw_steps):
            if not isinstance(rs, dict):
                continue
            domain = str(rs.get("domain", "")).strip()
            if domain not in available:          # never plan against a hidden/unknown module
                continue
            subq = str(rs.get("subq") or "").strip()
            if not subq:
                continue
            sid = (str(rs.get("id") or "").strip() or f"s{i + 1}")
            if sid in seen_ids:
                sid = f"{sid}_{i}"
            # acyclic-by-construction: dependencies may only point at earlier steps.
            deps = [str(d).strip() for d in (rs.get("depends_on") or []) if str(d).strip() in seen_ids]
            steps.append(PlanStep(id=sid, domain=domain, subq=subq, depends_on=deps))
            seen_ids.add(sid)
            if len(steps) >= self.settings.max_plan_steps:
                break
        if not steps:
            return None
        synthesis = str(data.get("synthesis") or "Answer from the findings; cite every claim.")
        return Plan(steps=steps, synthesis=synthesis, mode="llm")


__all__ = ["Planner", "Plan", "PlanStep"]
