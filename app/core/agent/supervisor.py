"""Manifest-driven SUPERVISOR / router — decides WHO answers, not WHAT to answer.

In the agent graph this backs ``route_node``. Its only job: given a question,
pick the capability module(s) best suited to handle it. It is deliberately dumb
and fast — the real work happens later in the specialists.

The key design property: the supervisor routes using ONLY the ``routing_hints``
each module declares in its manifest. It never hardcodes "if question mentions
threat actors -> aci". So when you drop in a new module with its own hints, it
becomes routable here WITH NO EDIT to this file. That "no core edit to add a
feature" rule is the whole point of the manifest system.

Strategies (``routing_strategy`` / ``router_mode``):
  * keyword — token overlap between the question and each module's hints + a bonus
    for an exact intent-phrase match. Zero infra, but brittle and does NOT scale to
    many apps (you'd hand-write keywords for every one).
  * semantic — EMBEDDING similarity between the query and each module's
    natural-language PROFILE (display name + description + hints + examples + tool
    names/descriptions). No keywords; scales to hundreds of apps (Composio-style).
    Module profile vectors are embedded ONCE and cached.
  * hybrid (default) — keyword when it has a signal, semantic when keyword is
    silent. Keeps the precise keyword routes while gaining semantic recall for the
    long tail (the case that matters when you bolt on a big tool catalog).
  * llm (``router_mode=llm`` + a real model) — ask the LLM to pick the subset.
With one module it always routes there; with many it ranks and can fan out up to
``max_fanout`` modules (cross-pillar questions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from app.core.llm.base import ChatMessage, Lane
from app.core.security.context import SecurityContext

_WORD = re.compile(r"[a-z0-9]+")
# Stopwords stripped before matching — these words carry no routing signal, so
# counting them would blur the difference between modules.
_STOP = {"the", "a", "an", "is", "are", "what", "which", "our", "my", "of", "to",
         "in", "on", "for", "and", "or", "how", "do", "does", "we", "i", "any"}

# Cache of module-PROFILE embedding vectors, keyed by (module_id, embedder
# provider). Profiles are stable, so we embed each app once and reuse the vector
# across turns — this is what makes semantic routing cheap even with many apps.
_MODULE_VEC_CACHE: dict[tuple, list] = {}


def _toks(text: str) -> set[str]:
    """Lowercase word set with stopwords and 1-char tokens removed."""
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def _cos(a, b) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    return float(va @ vb / (na * nb)) if na and nb else 0.0


@dataclass
class RouteResult:
    """What the supervisor returns. ``modules`` is the decision; the rest is
    explainability (surfaced by /route/preview and logged for debugging)."""

    modules: list[str]                              # chosen module ids, best first
    scores: dict[str, float] = field(default_factory=dict)  # per-module heuristic score
    mode: str = "heuristic"                         # "heuristic" | "llm" | "none"
    fallback: bool = False                          # True => nothing matched, used default


class Supervisor:
    def __init__(self, registry, llm, settings, *, default_module: str = "reports",
                 max_fanout: int = 2, embedder=None) -> None:
        self.registry = registry
        self.llm = llm
        self.settings = settings
        self.default_module = default_module   # where to route when nothing matches
        self.max_fanout = max_fanout           # cap on how many modules one question hits
        self.embedder = embedder               # for semantic routing (None => keyword only)

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

    def _module_profile(self, module) -> str:
        """The natural-language PROFILE we embed for semantic routing. Note the
        routing_hints are folded in here as DESCRIPTIVE TEXT (not exact-match
        keywords) — alongside the description and the tool names/descriptions — so
        a module is matched by meaning, not by literal token overlap. This is what
        lets you bolt on a 300-app catalog (Composio) without writing keywords."""
        m = module.manifest
        hints = " ".join(i for h in m.routing_hints for i in h.intents)
        examples = " ".join(e for h in m.routing_hints for e in h.examples)
        tools = " ".join(f"{t.name} {t.description}" for t in module.tools.values())
        return f"{m.display_name}. {m.description}. {hints}. {examples}. {tools}".strip()

    async def _semantic(self, question: str, module_ids: list[str]) -> dict[str, float]:
        """Score modules by EMBEDDING similarity between the query and each module's
        profile. Profile vectors are cached per (module, embedder) so each app is
        embedded once. Never raises — on any failure returns {} and the caller
        falls back to keyword routing."""
        if self.embedder is None:
            return {}
        try:
            qv = await self.embedder.embed_query(question)
            scores: dict[str, float] = {}
            missing_ids: list[str] = []
            missing_text: list[str] = []
            for mid in module_ids:
                mod = self.registry.module(mid)
                if not mod:
                    continue
                key = (mid, getattr(self.embedder, "provider", "?"))
                vec = _MODULE_VEC_CACHE.get(key)
                if vec is not None:
                    scores[mid] = _cos(qv, vec)
                else:
                    missing_ids.append(mid)
                    missing_text.append(self._module_profile(mod))
            if missing_text:
                vecs = await self.embedder.embed(missing_text)
                for mid, vec in zip(missing_ids, vecs, strict=False):
                    _MODULE_VEC_CACHE[(mid, getattr(self.embedder, "provider", "?"))] = vec
                    scores[mid] = _cos(qv, vec)
            return scores
        except Exception:
            return {}

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

        # Keyword scores — cheap, and the explainability payload even when another
        # strategy makes the final pick.
        kw = self._heuristic(question, view.routing)

        # LLM router (real provider) wins when configured.
        if self.settings.router_mode == "llm" and getattr(self.llm, "provider", "") != "deterministic":
            llm_pick = await self._llm_route(question, available)
            if llm_pick:
                return RouteResult(modules=llm_pick, scores=kw, mode="llm")

        strategy = getattr(self.settings, "routing_strategy", "hybrid")
        sem = (await self._semantic(question, available)
               if strategy in ("semantic", "hybrid") and self.embedder is not None else {})

        # Pure semantic: rank by embedding similarity only — no keywords at all
        # (the right mode for a large, keyword-less catalog like Composio).
        if strategy == "semantic" and sem:
            ranked = sorted(sem, key=lambda m: sem[m], reverse=True)[: self.max_fanout]
            return RouteResult(modules=ranked, scores=sem, mode="semantic")

        # Hybrid (default): keyword decides when it has a signal (semantic only
        # breaks ties, so precise keyword routes are preserved exactly); semantic
        # decides when keyword is silent — the long-tail / big-catalog case.
        if strategy == "hybrid":
            if kw:
                ranked = sorted(kw, key=lambda m: (kw[m], sem.get(m, 0.0)), reverse=True)[: self.max_fanout]
                return RouteResult(modules=ranked, scores=kw, mode="hybrid")
            if sem:
                ranked = sorted(sem, key=lambda m: sem[m], reverse=True)[: self.max_fanout]
                return RouteResult(modules=ranked, scores=sem, mode="hybrid-semantic")

        # Keyword strategy (or no embedder available): top keyword matches.
        if kw:
            ranked = sorted(kw, key=lambda m: kw[m], reverse=True)[: self.max_fanout]
            return RouteResult(modules=ranked, scores=kw, mode="heuristic")

        # Nothing matched at all -> default general module (best-effort answer).
        fb = self.default_module if self.default_module in available else available[0]
        return RouteResult(modules=[fb], scores=kw, mode="heuristic", fallback=True)


__all__ = ["Supervisor", "RouteResult"]
