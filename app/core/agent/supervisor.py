"""Manifest-driven SUPERVISOR / router — decides WHO answers, not WHAT to answer.

In the agent graph this backs ``route_node`` (heuristic mode) and the planner's
domain selection (planner mode). Its only job: given a question, pick the
capability module(s) best suited to handle it. It is deliberately fast — the real
work happens later in the specialists.

ROUTING IS DYNAMIC — BY MEANING, NOT KEYWORDS. The supervisor decides using each
module's natural-language PROFILE (``display_name`` + ``description`` + every tool's
name + description). There are NO hand-maintained intent keywords: a question whose
*meaning* matches a module routes there even when it shares no literal words with
the module's text. This is the property that makes routing robust in production —
the old keyword/``routing_hints`` approach silently failed whenever a real query
needed a capability but used different words than the curated phrases.

Two strategies (``router_mode`` / ``routing_strategy``):
  * semantic (default) — EMBEDDING similarity between the query and each module's
    profile. The default embedder is the deterministic, offline blake2 hashed
    bag-of-words (rag/embeddings.py), so semantic routing works with NO model and
    NO network; swap in a real embedder for sharper recall. Profile vectors are
    embedded ONCE and cached per (module, embedder).
  * llm (``router_mode=llm`` + a real model) — hand the model the question + the
    module ids and let it pick the relevant subset. Best judgement; needs a model.

With one entitled module it always routes there. With many it ranks by similarity
and fans out to the top ``max_fanout`` modules, so a genuinely cross-domain
question reaches both relevant specialists (e.g. "our biggest exposure AND which
actor weaponizes it" -> easm + aci) while a focused one is led by its best match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from app.core.llm.base import ChatMessage, Lane
from app.core.security.context import SecurityContext

# Cache of module-PROFILE embedding vectors, keyed by (module_id, embedder
# provider). Profiles are stable, so we embed each module once and reuse the
# vector across turns — this is what makes semantic routing cheap with many apps.
_MODULE_VEC_CACHE: dict[tuple, list] = {}


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
    scores: dict[str, float] = field(default_factory=dict)  # per-module similarity score
    mode: str = "semantic"                          # "semantic" | "llm" | "single" | "fallback" | "none"
    fallback: bool = False                          # True => no signal, used default


class Supervisor:
    def __init__(self, registry, llm, settings, *, default_module: str = "reports",
                 max_fanout: int = 2, embedder=None) -> None:
        self.registry = registry
        self.llm = llm
        self.settings = settings
        self.default_module = default_module   # where to route when there is no signal at all
        self.max_fanout = max_fanout           # cap on how many modules one question hits
        self.embedder = embedder               # semantic routing (default: deterministic embedder)

    def _module_profile(self, module) -> str:
        """The natural-language PROFILE we embed for semantic routing — built purely
        from declarative module metadata: display name + description + every tool's
        name and description. No curated keywords. A module becomes routable simply
        by having a clear description and well-described tools, which is also what
        the LLM and the specialists rely on — one source of truth for 'what this
        module is for'."""
        m = module.manifest
        tools = " ".join(f"{t.name} {t.description}" for t in module.tools.values())
        return f"{m.display_name}. {m.description}. {tools}".strip()

    async def _semantic(self, question: str, module_ids: list[str]) -> dict[str, float]:
        """Score modules by EMBEDDING similarity between the query and each module's
        profile. Profile vectors are cached per (module, embedder) so each module is
        embedded once. Never raises — on any failure returns {} and the caller falls
        back to the default module."""
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
        """LLM routing: hand the model the question + the list of module ids (and
        their one-line descriptions) and ask for the relevant subset. Runs on the
        FAST lane (routing is cheap). Any failure returns [] so the caller falls
        back to semantic — the router must never hard-fail a turn."""
        cards = []
        for mid in module_ids:
            mod = self.registry.module(mid)
            desc = mod.manifest.description if mod else ""
            cards.append(f"- {mid}: {desc}")
        sys = (
            "You are a router for a security-intelligence assistant. Given a user "
            "question and the capability modules below, reply with a comma-separated "
            "subset (most relevant first) that should handle it. Choose by the MEANING "
            "of the question, not shared words. Use ONE module unless the question "
            "genuinely spans several. Reply ONLY with module ids.\n\n" + "\n".join(cards)
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
        """The entry point ``route_node`` (and the planner's heuristic fallback)
        call. Note ``capability_view(sc)`` — the supervisor only ever considers
        modules THIS org/user is allowed to see (RBAC + license filtering happen in
        the registry), so routing can never leak the existence of a module a tenant
        isn't entitled to.

        Decision order: a single entitled module routes trivially; an LLM router
        wins when configured; otherwise semantic similarity ranks the modules (the
        default, deterministic, offline path). Only when there is NO embedder at all
        do we fall back to the default module."""
        view = self.registry.capability_view(sc)
        available = list(dict.fromkeys(view.module_ids))  # de-dupe, preserve order
        if not available:
            return RouteResult(modules=[], mode="none")
        if len(available) == 1:
            return RouteResult(modules=available, scores={available[0]: 1.0}, mode="single")

        # LLM router (real provider) wins when configured.
        if self.settings.router_mode == "llm" and getattr(self.llm, "provider", "") != "deterministic":
            llm_pick = await self._llm_route(question, available)
            if llm_pick:
                return RouteResult(modules=llm_pick, mode="llm")

        # Semantic routing — the default. Works offline via the deterministic
        # embedder; sharper with a real one. Ranks by meaning; fans out to top N.
        sem = await self._semantic(question, available)
        if sem:
            ranked = sorted(sem, key=lambda m: sem[m], reverse=True)[: self.max_fanout]
            return RouteResult(modules=ranked, scores=sem, mode="semantic")

        # No embedder available at all -> best-effort default module.
        fb = self.default_module if self.default_module in available else available[0]
        return RouteResult(modules=[fb], mode="fallback", fallback=True)


__all__ = ["Supervisor", "RouteResult"]
