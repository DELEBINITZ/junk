"""Vector store contract + in-memory backend.

WHAT A VECTOR STORE IS: the database that holds every chunk's embedding vector and
answers "given this query vector, return the most similar chunks". It's the
retrieval half of RAG — chunking/embedding fill it, search reads from it.

THE SECURITY RULE THAT MATTERS MOST — multi-tenant isolation:
This is a multi-TENANT system: many organizations' data live in the same
collection. The one boundary that must never leak is org A seeing org B's chunks.
So tenant isolation is enforced *here, at the vector layer*: every ``search``
REQUIRES an ``org_id`` and only ever returns points tagged with that org. Crucially
``org_id`` comes from the TRUSTED request context (derived from the verified token,
carried in ToolContext) — NEVER from the user's query text or any tool argument, so
a user cannot ask their way into another org's data. This is the single most
important isolation boundary for RAG (blueprint §11). The Qdrant backend enforces
the identical rule with a mandatory ``must`` filter at the database itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, Field

from app.core.contracts import Chunk


class VectorRecord(BaseModel):
    # One stored row: the embedding ``vector`` plus its payload (text + provenance).
    # ``org_id`` is the mandatory tenant tag every record is stamped with at write
    # time; search filters on it to keep tenants apart.
    id: str
    org_id: str            # owning tenant — required, set from trusted context
    vector: list[float]    # the embedding (must match the collection's dim)
    text: str
    source: str = ""
    doc_id: str = ""
    title: str = ""
    section: str = ""
    published_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchFilters(BaseModel):
    # OPTIONAL, user-influenced narrowing applied ON TOP of the mandatory org_id
    # filter — never a replacement for it. org_id is enforced separately and is not
    # a field here, precisely so it can't be overridden via these filters.
    source: str | None = None
    date_from: str | None = None  # ISO; inclusive lower bound on published_at (recency filtering)
    date_to: str | None = None    # ISO; inclusive upper bound
    doc_ids: list[str] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)   # arbitrary metadata.* equality matches


class VectorStore(Protocol):
    # The backend-agnostic interface (Protocol = duck-typed). InMemoryVectorStore
    # below and QdrantVectorStore both implement it, so the pipeline is indifferent
    # to which is wired. Note ``org_id`` is a REQUIRED keyword on the read methods.
    async def ensure_collection(self, collection: str, dim: int) -> None: ...

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> int: ...

    async def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        *,
        org_id: str,                                # mandatory tenant scope
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[Chunk]: ...

    async def delete_by_doc(self, collection: str, org_id: str, doc_id: str) -> int: ...

    async def count(self, collection: str, org_id: str | None = None) -> int: ...

    async def aclose(self) -> None: ...


def _passes_filters(rec: VectorRecord, f: SearchFilters | None) -> bool:
    # Apply the OPTIONAL SearchFilters to one record (the in-memory equivalent of
    # Qdrant's payload filter). Note this does NOT check org_id — that hard tenant
    # check happens separately in search(), so it can never be accidentally skipped
    # by an empty filter object.
    if f is None:
        return True
    if f.source and rec.source != f.source:
        return False
    if f.doc_ids and rec.doc_id not in f.doc_ids:
        return False
    # Lexicographic string compare works as a date compare because the timestamps
    # are zero-padded ISO-8601 (the recency window from filters.py).
    if f.date_from and (rec.published_at or "") < f.date_from:
        return False
    if f.date_to and rec.published_at and rec.published_at > f.date_to:
        return False
    for k, v in (f.extra or {}).items():
        if rec.metadata.get(k) != v:
            return False
    return True


class InMemoryVectorStore:
    """Brute-force cosine search. Fine for dev/tests and small corpora.

    No real index — it scores the query against EVERY record in the collection.
    Simple and dependency-free; swap to the Qdrant backend for production scale."""

    backend = "memory"

    def __init__(self) -> None:
        # collection name -> {record id -> record}. A nested dict IS the "database".
        self._cols: dict[str, dict[str, VectorRecord]] = {}
        self._mat: dict[str, np.ndarray | None] = {}  # cache (invalidated on write)

    async def ensure_collection(self, collection: str, dim: int) -> None:
        # Create the collection if absent. ``dim`` is ignored here (no fixed-width
        # index to allocate); the Qdrant backend does use it.
        self._cols.setdefault(collection, {})

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> int:
        col = self._cols.setdefault(collection, {})
        for r in records:
            # Refuse to store an untagged record — a chunk with no org_id could be
            # returned to ANY tenant, so this guard protects isolation at write time.
            if not r.org_id:
                raise ValueError("VectorRecord.org_id is required (tenant isolation)")
            col[r.id] = r        # upsert: same id overwrites
        self._mat[collection] = None   # invalidate any cached matrix
        return len(records)

    async def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        *,
        org_id: str,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[Chunk]:
        # Fail closed: a search without a tenant scope is a bug, never a "return
        # everything". This mirrors the mandatory org_id filter the Qdrant backend
        # applies at the database.
        if not org_id:
            raise ValueError("search requires org_id (tenant isolation)")
        col = self._cols.get(collection, {})
        q = np.asarray(query_vector, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0     # query norm (||q||) for cosine
        scored: list[tuple[float, VectorRecord]] = []
        for rec in col.values():
            if rec.org_id != org_id:  # hard tenant boundary — skip other orgs' rows entirely
                continue
            if not _passes_filters(rec, filters):   # then apply the optional narrowing
                continue
            # Cosine similarity = dot(q, v) / (||q|| * ||v||): the angle-based
            # closeness that ranks chunks by semantic relevance to the query.
            v = np.asarray(rec.vector, dtype=np.float32)
            vn = float(np.linalg.norm(v)) or 1.0
            scored.append((float(np.dot(q, v) / (qn * vn)), rec))
        scored.sort(key=lambda t: t[0], reverse=True)   # best (most similar) first
        out: list[Chunk] = []
        # Keep the top_k and project each record into a Chunk (carrying score +
        # provenance) for the pipeline. This is the "fetch" that over-fetch-then-
        # rerank in pipeline.py builds on.
        for score, rec in scored[: max(0, top_k)]:
            out.append(Chunk(
                id=rec.id, text=rec.text, score=score, org_id=rec.org_id,
                source=rec.source, doc_id=rec.doc_id, title=rec.title,
                section=rec.section, published_at=rec.published_at, metadata=rec.metadata,
            ))
        return out

    async def delete_by_doc(self, collection: str, org_id: str, doc_id: str) -> int:
        # Delete is also org-scoped: only rows matching BOTH this org and this doc
        # are removed, so one tenant can never purge another's data.
        col = self._cols.get(collection, {})
        ids = [i for i, r in col.items() if r.org_id == org_id and r.doc_id == doc_id]
        for i in ids:
            col.pop(i, None)
        if ids:
            self._mat[collection] = None
        return len(ids)

    async def count(self, collection: str, org_id: str | None = None) -> int:
        # Total rows, or rows for one tenant when org_id is given (admin/metrics use).
        col = self._cols.get(collection, {})
        if org_id is None:
            return len(col)
        return sum(1 for r in col.values() if r.org_id == org_id)

    async def aclose(self) -> None:
        return None


__all__ = ["VectorRecord", "SearchFilters", "VectorStore", "InMemoryVectorStore"]
