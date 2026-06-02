"""Long-term entity memory — the seam for the shared knowledge graph.

v1 ships a NoOp (chat memory is enough for report Q&A). When the platform grows
cross-pillar (EASM↔ACI↔Brand), flip ``kg_provider=zep`` to back this with Zep +
Graphiti — a temporal entity graph, strictly tenant-isolated. The agent gets a
``recall_memory`` core tool only when a real KG is configured.
"""

from __future__ import annotations

from typing import Protocol


class KnowledgeGraph(Protocol):
    provider: str

    async def add_observation(self, org_id: str, user_id: str, text: str, metadata: dict | None = None) -> None: ...
    async def recall(self, org_id: str, user_id: str, query: str, *, limit: int = 5) -> list[str]: ...
    async def aclose(self) -> None: ...


class NoOpKnowledgeGraph:
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
        from zep_python.client import AsyncZep  # lazy

        self._client = AsyncZep(api_key=api_key, base_url=api_url or None)

    @staticmethod
    def _ns(org_id: str, user_id: str) -> str:
        return f"{org_id}::{user_id}"

    async def add_observation(self, org_id, user_id, text, metadata=None) -> None:
        try:
            await self._client.memory.add(
                session_id=self._ns(org_id, user_id),
                messages=[{"role": "system", "content": text, "metadata": metadata or {}}],
            )
        except Exception:
            return None

    async def recall(self, org_id, user_id, query, *, limit=5) -> list[str]:
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
    if settings.kg_provider == "zep" and settings.zep_api_url:
        return ZepKnowledgeGraph(settings.zep_api_url, settings.zep_api_key)
    return NoOpKnowledgeGraph()


__all__ = ["KnowledgeGraph", "NoOpKnowledgeGraph", "ZepKnowledgeGraph", "build_kg"]
