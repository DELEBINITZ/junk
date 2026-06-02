"""Long-term entity knowledge graph (plan §9.3).

NoOp by default; Zep adapter when KG_PROVIDER=zep (lazy `zep_python`). Strictly
org+user scoped — a cross-org leak here is worse than a chat leak. Modules
contribute entity/edge types (OntologyContribution); this is the cross-pillar
join that later powers "what's my biggest risk and who's behind it?".

When enabled, a `recall_memory` core tool exposes `search` to the agent.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from app.config import settings

logger = logging.getLogger(__name__)


class KnowledgeGraph(Protocol):
    def add_facts(self, org_id: str, user_id: str, facts: list[str]) -> None: ...
    def search(self, org_id: str, user_id: str, query: str, limit: int = 5) -> list[str]: ...


class NoOpKnowledgeGraph:
    def add_facts(self, org_id: str, user_id: str, facts: list[str]) -> None:
        return None

    def search(self, org_id: str, user_id: str, query: str, limit: int = 5) -> list[str]:
        return []


class ZepKnowledgeGraph:  # pragma: no cover - needs a Zep server
    """Per-(org,user) graph in Zep. Session/graph id namespaced by org+user so
    memory never crosses tenants."""

    def __init__(self):
        from zep_python.client import Zep  # lazy

        self._client = Zep(api_key=settings.zep_api_key, base_url=settings.zep_api_url or None)

    def _gid(self, org_id: str, user_id: str) -> str:
        return f"{org_id}:{user_id}"

    def add_facts(self, org_id: str, user_id: str, facts: list[str]) -> None:
        try:
            for fact in facts:
                self._client.graph.add(user_id=self._gid(org_id, user_id), type="text", data=fact)
        except Exception as exc:
            logger.warning("kg.add_failed", extra={"error": str(exc)})

    def search(self, org_id: str, user_id: str, query: str, limit: int = 5) -> list[str]:
        try:
            results = self._client.graph.search(user_id=self._gid(org_id, user_id), query=query, limit=limit)
            return [getattr(e, "fact", str(e)) for e in (results.edges or [])]
        except Exception as exc:
            logger.warning("kg.search_failed", extra={"error": str(exc)})
            return []


_kg: KnowledgeGraph | None = None


def get_kg() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        provider = os.getenv("KG_PROVIDER", settings.kg_provider).lower()
        if provider == "zep":
            try:
                _kg = ZepKnowledgeGraph()
            except Exception as exc:  # pragma: no cover
                logger.warning("kg.zep_unavailable", extra={"error": str(exc)})
                _kg = NoOpKnowledgeGraph()
        else:
            _kg = NoOpKnowledgeGraph()
    return _kg


def reset_kg() -> None:
    global _kg
    _kg = None
