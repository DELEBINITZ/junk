"""Long-term entity memory — the seam for the shared knowledge graph (KG).

================================ MENTAL MODEL =============================
Two kinds of memory exist in this system. The chat store (conversations.py) is
"what was said". A KNOWLEDGE GRAPH is "what we now KNOW" — durable facts about
entities (assets, CVEs, threat actors, domains) and the relationships between
them, accumulated across conversations. Think of it as the agent's long-term
memory vs. its short-term transcript.

This file is deliberately a SEAM, not a full implementation. v1 ships a NoOp
(plain chat memory is enough for report Q&A), so the rest of the code can call
``kg.recall(...)`` unconditionally and simply get nothing back. When the platform
grows cross-pillar (EASM↔ACI↔Brand) and needs real entity recall, you flip
``kg_provider=zep`` to back this with Zep + Graphiti — a *temporal* entity graph
(facts have validity over time), strictly tenant-isolated. Only when a real KG is
configured does the agent expose a ``recall_memory`` core tool.

The Protocol below is the contract every backend honors; ``build_kg`` is the
factory that picks one from settings (the same config-driven pattern as the chat
store). Note ``org_id`` + ``user_id`` on every call — tenant isolation again.
===========================================================================
"""

from __future__ import annotations

from typing import Protocol


class KnowledgeGraph(Protocol):
    """Interface for long-term entity memory. ``add_observation`` writes a fact;
    ``recall`` fetches facts relevant to a query. ``provider`` names the backend
    ("none"/"zep") for health/diagnostics. Structural typing: any class with these
    methods qualifies, so swapping providers needs no base-class change."""

    provider: str

    async def add_observation(self, org_id: str, user_id: str, text: str, metadata: dict | None = None) -> None: ...
    async def recall(self, org_id: str, user_id: str, query: str, *, limit: int = 5) -> list[str]: ...
    async def aclose(self) -> None: ...


class NoOpKnowledgeGraph:
    """The DEFAULT backend: a no-op. Writes are dropped and recalls return nothing,
    so the agent runs with no KG infrastructure at all while every call site stays
    identical. This is the "Null Object" pattern — absence is modeled as a real
    object so callers never need ``if kg is not None`` branches."""

    provider = "none"

    async def add_observation(self, org_id, user_id, text, metadata=None) -> None:
        return None

    async def recall(self, org_id, user_id, query, *, limit=5) -> list[str]:
        return []

    async def aclose(self) -> None:
        return None


class ZepKnowledgeGraph:
    """Zep-backed temporal KG. Namespaces every session by ``org:user`` so a
    tenant can never read another's graph. Lazy-imports ``zep_python``."""

    provider = "zep"

    def __init__(self, api_url: str, api_key: str) -> None:
        # Lazy import: the dependency is only needed when this backend is actually
        # selected, so the default NoOp deployment ships without zep_python.
        from zep_python.client import AsyncZep  # lazy

        self._client = AsyncZep(api_key=api_key, base_url=api_url or None)

    @staticmethod
    def _ns(org_id: str, user_id: str) -> str:
        # THE isolation mechanism for this backend: the graph namespace key bakes
        # the tenant (and user) into Zep's session id. A query for one (org, user)
        # physically cannot touch another's namespace. org_id comes from the
        # verified token upstream, never from anything user-controlled.
        return f"{org_id}::{user_id}"

    async def add_observation(self, org_id, user_id, text, metadata=None) -> None:
        # Memory writes are best-effort: a KG outage must never break a chat turn,
        # so any error is swallowed (the answer still works, we just learned less).
        try:
            await self._client.memory.add(
                session_id=self._ns(org_id, user_id),
                messages=[{"role": "system", "content": text, "metadata": metadata or {}}],
            )
        except Exception:
            return None

    async def recall(self, org_id, user_id, query, *, limit=5) -> list[str]:
        # Same best-effort stance on reads: degrade to "no memory" rather than fail.
        try:
            res = await self._client.memory.search(
                session_id=self._ns(org_id, user_id), text=query, limit=limit
            )
            return [r.message.content for r in (res or []) if getattr(r, "message", None)]
        except Exception:
            return []

    async def aclose(self) -> None:
        return None


def build_kg(settings) -> KnowledgeGraph:
    """Factory: choose the KG backend from settings. Real Zep only if it is both
    requested (``kg_provider=zep``) AND has a URL configured; otherwise fall back
    to the NoOp so a half-configured deployment degrades safely instead of
    crashing at boot. Called once during bootstrap and stashed on CoreDeps.kg."""
    if settings.kg_provider == "zep" and settings.zep_api_url:
        return ZepKnowledgeGraph(settings.zep_api_url, settings.zep_api_key)
    return NoOpKnowledgeGraph()


__all__ = ["KnowledgeGraph", "NoOpKnowledgeGraph", "ZepKnowledgeGraph", "build_kg"]
