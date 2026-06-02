"""Ingestion service — the org-scoped write path into the retrieval corpus.

This is the "chunk -> index" tail of the ingestion pipeline (parsers.py did the
"parse"). It splits a document into overlapping CHUNKS and upserts them, via the
shared RAG pipeline (``deps.rag``), into the org's vector store — the same store
the agent's retrievers read at query time. Backs the ``/ingest`` endpoint and the
dev seeders.

SECURITY INVARIANT (the key line in this file): the destination tenant is always
``ctx.org_id`` — taken from the TRUSTED ToolContext (derived from the verified
token), NEVER from the request payload. So a caller can only ever write into its
own org's slice; it cannot inject data into another tenant's corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.core.contracts import IngestStats, ToolContext
from app.core.rag.chunking import chunk_document
from app.core.rag.pipeline import IndexItem


class IngestDocument(BaseModel):
    """One document to ingest, as the API/seeder supplies it. Note there is NO
    ``org_id`` field — tenancy comes from the trusted context at write time, so it
    is structurally impossible to target another org via the payload."""
    doc_id: str
    title: str = ""
    text: str
    source: str = "reports"
    published_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionService:
    """The write-side service. Holds ``deps`` so it can reach the shared RAG
    pipeline; one instance is built in bootstrap and reused."""

    def __init__(self, deps) -> None:
        self.deps = deps

    async def ingest_documents(
        self, ctx: ToolContext, collection: str, docs: Sequence[IngestDocument], *, chunk: bool = True
    ) -> IngestStats:
        """Chunk (optionally) and index a batch of documents into ``collection``
        for ``ctx.org_id``. RAG retrieves passages, so long docs are split into
        overlapping pieces; each piece becomes one ``IndexItem``. The per-chunk id
        ``{doc_id}::{ordinal}`` keeps a stable, document-grouped identity (so a
        later re-ingest or delete can find all pieces of one doc). ``chunk=False``
        indexes the whole text as a single item (used for already-small inputs)."""
        items: list[IndexItem] = []
        for d in docs:
            if chunk:
                # chunk_document yields ordered pieces (with section labels); each
                # one is indexed separately so retrieval can return just the
                # relevant passage rather than the whole document.
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
        # The pipeline embeds + upserts the items, scoped to this org. ``n`` is the
        # number of chunks actually written.
        n = await self.deps.rag.index(collection, ctx.org_id, items)
        return IngestStats(documents=len(docs), chunks=n)

    async def ingest_raw(
        self, ctx: ToolContext, collection: str, *, doc_id: str, title: str,
        data: bytes, content_type: str = "", filename: str = "",
        source: str = "reports", published_at: str | None = None,
    ) -> IngestStats:
        """The file-upload entry point: PARSE raw bytes to text (parsers.py), then
        hand off to ``ingest_documents``. This is the full parse -> chunk -> index
        path for a single uploaded file."""
        from app.core.ingestion.parsers import parse

        text = parse(data, content_type=content_type, filename=filename)
        return await self.ingest_documents(
            ctx, collection,
            [IngestDocument(doc_id=doc_id, title=title, text=text, source=source, published_at=published_at)],
        )

    async def delete_document(self, ctx: ToolContext, collection: str, doc_id: str) -> int:
        """Remove every chunk of one document from the org's corpus. Again scoped
        to ``ctx.org_id``, so a tenant can only delete from its own slice."""
        return await self.deps.rag.store.delete_by_doc(collection, ctx.org_id, doc_id)


__all__ = ["IngestDocument", "IngestionService"]
