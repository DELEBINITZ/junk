"""Event-driven ingestion seam.

In production an external cron/event source pushes updates; this in-process
``EventBus`` + ``IngestionConnector`` shape is how those land in the corpus
(and later the KG) without touching the chat path. A connector normalizes a
source event and writes to sinks; the bus routes events to connectors by source.
Durable/at-scale ingestion is delegated to the Temporal worker seam.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.contracts import Chunk, IngestStats, SourceEvent
from app.core.rag.pipeline import IndexItem


class RagKgSinks:
    """Sinks that write normalized chunks into the org-scoped vector corpus and
    (optionally) observations into the knowledge graph."""

    def __init__(self, deps, collection: str) -> None:
        self.deps = deps
        self.collection = collection

    async def write_chunks(self, chunks: Sequence[Chunk]) -> int:
        by_org: dict[str, list[IndexItem]] = {}
        for c in chunks:
            by_org.setdefault(c.org_id, []).append(IndexItem(
                id=c.id, text=c.text, source=c.source, doc_id=c.doc_id, title=c.title,
                section=c.section, published_at=c.published_at, metadata=c.metadata,
            ))
        total = 0
        for org_id, items in by_org.items():
            total += await self.deps.rag.index(self.collection, org_id, items)
        return total

    async def write_graph(self, org_id: str, nodes: Sequence[dict], edges: Sequence[dict]) -> None:
        for n in nodes:
            await self.deps.kg.add_observation(org_id, "ingest", str(n), {"kind": "node"})


class EventBus:
    def __init__(self, deps) -> None:
        self.deps = deps
        self._connectors: dict[str, list] = {}

    def register(self, connector) -> None:
        self._connectors.setdefault(connector.source, []).append(connector)

    async def publish(self, event: SourceEvent, *, collection: str) -> IngestStats:
        stats = IngestStats()
        sinks = RagKgSinks(self.deps, collection)
        for connector in self._connectors.get(event.source, []):
            s = await connector.handle(event, sinks)
            stats.documents += s.documents
            stats.chunks += s.chunks
            stats.errors += s.errors
        return stats


__all__ = ["RagKgSinks", "EventBus"]
