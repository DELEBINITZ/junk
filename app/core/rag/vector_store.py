"""Vector store contract + in-memory backend.

**Tenant isolation is enforced at the vector layer**: every ``search`` REQUIRES
an ``org_id`` and only ever returns points tagged with that org. This is the
single most important isolation boundary for RAG (blueprint §11). The Qdrant
backend enforces the same with a mandatory ``must`` filter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, Field

from app.core.contracts import Chunk


class VectorRecord(BaseModel):
    id: str
    org_id: str
    vector: list[float]
    text: str
    source: str = ""
    doc_id: str = ""
    title: str = ""
    section: str = ""
    published_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchFilters(BaseModel):
    source: str | None = None
    date_from: str | None = None  # ISO; inclusive lower bound on published_at
    date_to: str | None = None    # ISO; inclusive upper bound
    doc_ids: list[str] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class VectorStore(Protocol):
    async def ensure_collection(self, collection: str, dim: int) -> None: ...

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> int: ...

    async def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        *,
        org_id: str,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[Chunk]: ...

    async def delete_by_doc(self, collection: str, org_id: str, doc_id: str) -> int: ...

    async def count(self, collection: str, org_id: str | None = None) -> int: ...

    async def aclose(self) -> None: ...


def _passes_filters(rec: VectorRecord, f: SearchFilters | None) -> bool:
    if f is None:
        return True
    if f.source and rec.source != f.source:
        return False
    if f.doc_ids and rec.doc_id not in f.doc_ids:
        return False
    if f.date_from and (rec.published_at or "") < f.date_from:
        return False
    if f.date_to and rec.published_at and rec.published_at > f.date_to:
        return False
    for k, v in (f.extra or {}).items():
        if rec.metadata.get(k) != v:
            return False
    return True


class InMemoryVectorStore:
    """Brute-force cosine search. Fine for dev/tests and small corpora."""

    backend = "memory"

    def __init__(self) -> None:
        self._cols: dict[str, dict[str, VectorRecord]] = {}
        self._mat: dict[str, np.ndarray | None] = {}  # cache (invalidated on write)

    async def ensure_collection(self, collection: str, dim: int) -> None:
        self._cols.setdefault(collection, {})

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> int:
        col = self._cols.setdefault(collection, {})
        for r in records:
            if not r.org_id:
                raise ValueError("VectorRecord.org_id is required (tenant isolation)")
            col[r.id] = r
        self._mat[collection] = None
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
        if not org_id:
            raise ValueError("search requires org_id (tenant isolation)")
        col = self._cols.get(collection, {})
        q = np.asarray(query_vector, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        scored: list[tuple[float, VectorRecord]] = []
        for rec in col.values():
            if rec.org_id != org_id:  # hard tenant boundary
                continue
            if not _passes_filters(rec, filters):
                continue
            v = np.asarray(rec.vector, dtype=np.float32)
            vn = float(np.linalg.norm(v)) or 1.0
            scored.append((float(np.dot(q, v) / (qn * vn)), rec))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[Chunk] = []
        for score, rec in scored[: max(0, top_k)]:
            out.append(Chunk(
                id=rec.id, text=rec.text, score=score, org_id=rec.org_id,
                source=rec.source, doc_id=rec.doc_id, title=rec.title,
                section=rec.section, published_at=rec.published_at, metadata=rec.metadata,
            ))
        return out

    async def delete_by_doc(self, collection: str, org_id: str, doc_id: str) -> int:
        col = self._cols.get(collection, {})
        ids = [i for i, r in col.items() if r.org_id == org_id and r.doc_id == doc_id]
        for i in ids:
            col.pop(i, None)
        if ids:
            self._mat[collection] = None
        return len(ids)

    async def count(self, collection: str, org_id: str | None = None) -> int:
        col = self._cols.get(collection, {})
        if org_id is None:
            return len(col)
        return sum(1 for r in col.values() if r.org_id == org_id)

    async def aclose(self) -> None:
        return None


__all__ = ["VectorRecord", "SearchFilters", "VectorStore", "InMemoryVectorStore"]
