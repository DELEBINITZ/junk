"""Ingestion service — the org-scoped write path into the retrieval corpus.

Backs the ``/ingest`` endpoint and dev seeders. Embedding + upsert go through the
shared RAG pipeline (``deps.rag``); ``org_id`` always comes from the trusted
context, never the payload, so a tenant can only ever write into its own slice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.core.contracts import IngestStats, ToolContext
from app.core.rag.chunking import chunk_document
from app.core.rag.pipeline import IndexItem


class IngestDocument(BaseModel):
    doc_id: str
    title: str = ""
    text: str
    source: str = "reports"
    published_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionService:
    def __init__(self, deps) -> None:
        self.deps = deps

    async def ingest_documents(
        self, ctx: ToolContext, collection: str, docs: Sequence[IngestDocument], *, chunk: bool = True
    ) -> IngestStats:
        items: list[IndexItem] = []
        for d in docs:
            if chunk:
                for piece in chunk_document(d.text):
                    items.append(IndexItem(
                        id=f"{d.doc_id}::{piece.ordinal}", text=piece.text, source=d.source,
                        doc_id=d.doc_id, title=d.title, section=piece.section,
                        published_at=d.published_at, metadata=d.metadata,
                    ))
            else:
                items.append(IndexItem(
                    id=f"{d.doc_id}::0", text=d.text, source=d.source, doc_id=d.doc_id,
                    title=d.title, published_at=d.published_at, metadata=d.metadata,
                ))
        n = await self.deps.rag.index(collection, ctx.org_id, items)
        return IngestStats(documents=len(docs), chunks=n)

    async def ingest_raw(
        self, ctx: ToolContext, collection: str, *, doc_id: str, title: str,
        data: bytes, content_type: str = "", filename: str = "",
        source: str = "reports", published_at: str | None = None,
    ) -> IngestStats:
        from app.core.ingestion.parsers import parse

        text = parse(data, content_type=content_type, filename=filename)
        return await self.ingest_documents(
            ctx, collection,
            [IngestDocument(doc_id=doc_id, title=title, text=text, source=source, published_at=published_at)],
        )

    async def delete_document(self, ctx: ToolContext, collection: str, doc_id: str) -> int:
        return await self.deps.rag.store.delete_by_doc(collection, ctx.org_id, doc_id)


__all__ = ["IngestDocument", "IngestionService"]
