"""Vector store contract (the Qdrant backend implements it).

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
the rule with a mandatory ``must`` filter at the database itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

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
    # The backend-agnostic interface (Protocol = duck-typed). QdrantVectorStore
    # implements it, so the pipeline is indifferent to the concrete backend. Note
    # ``org_id`` is a REQUIRED keyword on the read methods.
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


__all__ = ["VectorRecord", "SearchFilters", "VectorStore"]
