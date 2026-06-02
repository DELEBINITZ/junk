"""Manifest-driven supervisor / router.

Selects which capability module(s) should handle a question using ONLY the
``routing_hints`` in each org-visible module's manifest — so registering a new
module makes it routable with no edit here. v1 routes among one module (reports)
and behaves like a single agent; with N modules it ranks and can fan out. An
optional LLM mode upgrades selection on real providers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.llm.base import ChatMessage, Lane
from app.core.security.context import SecurityContext

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "is", "are", "what", "which", "our", "my", "of", "to",
         "in", "on", "for", "and", "or", "how", "do", "does", "we", "i", "any"}


def _toks(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


@dataclass
class RouteResult:
    modules: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    mode: str = "heuristic"
    fallback: bool = False


class Supervisor:
    def __init__(self, registry, llm, settings, *, default_module: str = "reports", max_fanout: int = 2) -> None:
        self.registry = registry
        self.llm = llm
        self.settings = settings
        self.default_module = default_module
        self.max_fanout = max_fanout

    def _heuristic(self, question: str, routing: list[tuple[str, object]]) -> dict[str, float]:
        q = _toks(question)
        scores: dict[str, float] = {}
        for module_id, hint in routing:
            hint_tokens = _toks(" ".join(hint.intents) + " " + " ".join(hint.examples))
            inter = len(q & hint_tokens)
            # also reward exact intent-phrase substring match
            phrase_bonus = sum(1.5 for it in hint.intents if it.lower() in question.lower())
            score = inter + phrase_bonus
            if score > 0:
                scores[module_id] = max(scores.get(module_id, 0.0), float(score))
        return scores

    async def _llm_route(self, question: str, module_ids: list[str]) -> list[str]:
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
            picked = [m.strip() for m in re.split(r"[,\s]+", resp.text) if m.strip() in module_ids]
            return picked[: self.max_fanout]
        except Exception:
            return []

    async def route(self, question: str, sc: SecurityContext) -> RouteResult:
        view = self.registry.capability_view(sc)
        available = list(dict.fromkeys(view.module_ids))  # preserve order, unique
        if not available:
            return RouteResult(modules=[], mode="none")

        scores = self._heuristic(question, view.routing)

        if self.settings.router_mode == "llm" and getattr(self.llm, "provider", "") != "deterministic":
            llm_pick = await self._llm_route(question, available)
            if llm_pick:
                return RouteResult(modules=llm_pick, scores=scores, mode="llm")

        if scores:
            ranked = sorted(scores, key=lambda m: scores[m], reverse=True)[: self.max_fanout]
            return RouteResult(modules=ranked, scores=scores, mode="heuristic")

        # Fallback: route to the default general-purpose module if visible, else first.
        fb = self.default_module if self.default_module in available else available[0]
        return RouteResult(modules=[fb], scores=scores, mode="heuristic", fallback=True)


__all__ = ["Supervisor", "RouteResult"]
