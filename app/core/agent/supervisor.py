"""Manifest-driven SUPERVISOR / router — decides WHO answers, not WHAT to answer.

In the agent graph this backs ``route_node``. Its only job: given a question,
pick the capability module(s) best suited to handle it. It is deliberately dumb
and fast — the real work happens later in the specialists.

The key design property: the supervisor routes using ONLY the ``routing_hints``
each module declares in its manifest. It never hardcodes "if question mentions
threat actors -> aci". So when you drop in a new module with its own hints, it
becomes routable here WITH NO EDIT to this file. That "no core edit to add a
feature" rule is the whole point of the manifest system.

Two modes:
  * heuristic (default) — token overlap between the question and each module's
    hints, plus a bonus for an exact intent-phrase match. Zero infra.
  * llm (``router_mode=llm`` + a real model) — ask the LLM to pick the subset.
With one module it always routes there (acts like a single agent); with many it
ranks and can fan out up to ``max_fanout`` modules (cross-pillar questions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.llm.base import ChatMessage, Lane
from app.core.security.context import SecurityContext

_WORD = re.compile(r"[a-z0-9]+")
# Stopwords stripped before matching — these words carry no routing signal, so
# counting them would blur the difference between modules.
_STOP = {"the", "a", "an", "is", "are", "what", "which", "our", "my", "of", "to",
         "in", "on", "for", "and", "or", "how", "do", "does", "we", "i", "any"}


def _toks(text: str) -> set[str]:
    """Lowercase word set with stopwords and 1-char tokens removed."""
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


@dataclass
class RouteResult:
    """What the supervisor returns. ``modules`` is the decision; the rest is
    explainability (surfaced by /route/preview and logged for debugging)."""

    modules: list[str]                              # chosen module ids, best first
    scores: dict[str, float] = field(default_factory=dict)  # per-module heuristic score
    mode: str = "heuristic"                         # "heuristic" | "llm" | "none"
    fallback: bool = False                          # True => nothing matched, used default


class Supervisor:
    def __init__(self, registry, llm, settings, *, default_module: str = "reports", max_fanout: int = 2) -> None:
        self.registry = registry
        self.llm = llm
        self.settings = settings
        self.default_module = default_module   # where to route when nothing matches
        self.max_fanout = max_fanout           # cap on how many modules one question hits

    def _heuristic(self, question: str, routing: list[tuple[str, object]]) -> dict[str, float]:
        """Score each module by similarity of the question to its routing hints.

        score = (# overlapping words between question and the module's
                 intents+examples) + 1.5 for each intent phrase that appears
                 verbatim in the question.
        The phrase bonus rewards strong signals like the literal text
        "threat actor" over incidental single-word overlaps.
        """
        q = _toks(question)
        scores: dict[str, float] = {}
        for module_id, hint in routing:
            hint_tokens = _toks(" ".join(hint.intents) + " " + " ".join(hint.examples))
            inter = len(q & hint_tokens)                                   # word overlap
            phrase_bonus = sum(1.5 for it in hint.intents if it.lower() in question.lower())
            score = inter + phrase_bonus
            if score > 0:
                scores[module_id] = max(scores.get(module_id, 0.0), float(score))
        return scores

    async def _llm_route(self, question: str, module_ids: list[str]) -> list[str]:
        """LLM routing: hand the model the question + the list of module ids and
        ask for the relevant subset. Runs on the FAST lane (routing is cheap).
        Any failure returns [] so the caller falls back to the heuristic — the
        router must never hard-fail a turn."""
        labels = ", ".join(module_ids)
        sys = (
            "You are a router. Given a user question and a list of capability "
            f"modules [{labels}], reply with a comma-separated subset (most "
            "relevant first) that should handle it. Reply ONLY with module ids."
        )
        try:
            resp = await self.llm.complete(
                [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=question)],
                lane=Lane.FAST,
            )
            # Keep only tokens that are real module ids; cap to max_fanout.
            picked = [m.strip() for m in re.split(r"[,\s]+", resp.text) if m.strip() in module_ids]
            return picked[: self.max_fanout]
        except Exception:
            return []

    async def route(self, question: str, sc: SecurityContext) -> RouteResult:
        """The entry point ``route_node`` calls. Note ``capability_view(sc)`` —
        the supervisor only ever considers modules THIS org/user is allowed to
        see (RBAC + license filtering happen in the registry), so routing can
        never leak the existence of a module a tenant isn't entitled to."""
        view = self.registry.capability_view(sc)
        available = list(dict.fromkeys(view.module_ids))  # de-dupe, preserve order
        if not available:
            return RouteResult(modules=[], mode="none")

        # Always compute heuristic scores — they're cheap and double as the
        # explainability payload even when the LLM makes the final pick.
        scores = self._heuristic(question, view.routing)

        # LLM mode (only with a real provider). If it returns a pick, use it.
        if self.settings.router_mode == "llm" and getattr(self.llm, "provider", "") != "deterministic":
            llm_pick = await self._llm_route(question, available)
            if llm_pick:
                return RouteResult(modules=llm_pick, scores=scores, mode="llm")

        # Heuristic mode (default): take the top-scoring modules, up to max_fanout.
        if scores:
            ranked = sorted(scores, key=lambda m: scores[m], reverse=True)[: self.max_fanout]
            return RouteResult(modules=ranked, scores=scores, mode="heuristic")

        # Nothing matched at all -> route to the default general module (if the
        # tenant can see it) so the turn still gets a best-effort answer.
        fb = self.default_module if self.default_module in available else available[0]
        return RouteResult(modules=[fb], scores=scores, mode="heuristic", fallback=True)


__all__ = ["Supervisor", "RouteResult"]
