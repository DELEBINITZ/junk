"""Qdrant vector store (``retrieval_backend=qdrant``).

This is the PRODUCTION backend: Qdrant is a real vector database with an ANN
(approximate nearest-neighbour) index, so similarity search stays fast over
millions of vectors instead of the brute-force scan the in-memory store does.

It implements the same VectorStore contract as the in-memory store, with the
production-grade reinforcement of the rules taught there:
  * the **mandatory org_id filter is applied at the database layer** as a ``must``
    match (see ``_filter``) — tenant isolation enforced by the DB itself, not just
    by application code, so no query path can bypass it;
  * PAYLOAD INDEXES on the filtered fields (org_id, source, doc_id, published_at)
    let Qdrant filter quickly without scanning every point — like a column index
    in a SQL DB, this is what keeps filtered vector search fast;
  * datetime range filters fix the "last year returns old data" bug at the DB
    (dropping out-of-window points before ranking) instead of in post-ranking.
Lazy-imports ``qdrant-client`` so the dependency is only needed when this backend
is actually selected.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from app.core.contracts import Chunk
from app.core.rag.vector_store import SearchFilters, VectorRecord

# Fixed namespace UUID used to derive stable point ids from arbitrary string ids.
_NS = uuid.UUID("00000000-0000-0000-0000-0000000000aa")


def _point_id(rid: str) -> str:
    # Qdrant point ids must be a UUID or an int. If our record id already IS a
    # UUID, use it as-is; otherwise derive a DETERMINISTIC uuid5 from it so the
    # same record id always maps to the same point (re-ingest overwrites, not
    # duplicates). The real string id is also kept in the payload as ``rid``.
    try:
        return str(uuid.UUID(rid))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(_NS, rid))


class QdrantVectorStore:
    backend = "qdrant"

    def __init__(self, url: str, api_key: str = "", timeout: float = 30.0) -> None:
        from qdrant_client import AsyncQdrantClient  # lazy import: only when this backend is used

        self._client = AsyncQdrantClient(url=url, api_key=api_key or None, timeout=timeout)

    async def ensure_collection(self, collection: str, dim: int) -> None:
        from qdrant_client import models as qm

        exists = await self._client.collection_exists(collection)
        if not exists:
            # Create the collection sized to the embedder's ``dim`` and configured
            # for COSINE distance — must match how vectors are produced/normalized
            # in embeddings.py, or similarity scores would be meaningless.
            await self._client.create_collection(
                collection_name=collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
        # Build a payload index on each field we filter by, so those filters are
        # served by an index instead of a full scan. KEYWORD = exact-match fields;
        # DATETIME enables the published_at range queries. Idempotent: creating an
        # index that already exists just raises and is ignored.
        for field, schema in [
            ("org_id", qm.PayloadSchemaType.KEYWORD),         # the tenant filter — indexed for speed
            ("source", qm.PayloadSchemaType.KEYWORD),
            ("doc_id", qm.PayloadSchemaType.KEYWORD),
            ("published_at", qm.PayloadSchemaType.DATETIME),  # powers recency range filtering
            ("customer_tags", qm.PayloadSchemaType.KEYWORD),  # shared-corpus allow-list (array); Match + IsEmpty
        ]:
            try:
                await self._client.create_payload_index(collection, field_name=field, field_schema=schema)
            except Exception:  # noqa: BLE001 - already exists
                pass

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> int:
        from qdrant_client import models as qm

        points = []
        for r in records:
            # Same write-time guard as the in-memory store: never persist a chunk
            # without its tenant tag, or it could later match any org's search.
            if not r.org_id:
                raise ValueError("VectorRecord.org_id is required (tenant isolation)")
            # A Qdrant "point" = the vector (what ANN search compares) + a payload
            # (the metadata we filter on and return). org_id lives in the payload
            # so the mandatory filter below can match it.
            points.append(qm.PointStruct(
                id=_point_id(r.id),
                vector=r.vector,
                payload={
                    "rid": r.id, "org_id": r.org_id, "text": r.text, "source": r.source,
                    "doc_id": r.doc_id, "title": r.title, "section": r.section,
                    "published_at": r.published_at, "metadata": r.metadata,
                },
            ))
        if points:
            await self._client.upsert(collection_name=collection, points=points)
        return len(points)

    def _filter(self, org_id: str, f: SearchFilters | None, visibility: str = "tenant"):
        """Build the Qdrant filter. THE security-critical clause is the FIRST one,
        and it has two modes:

        * ``visibility="tenant"`` (DEFAULT) — ``org_id`` is in ``must`` (logical AND),
          so every search is HARD-scoped to one tenant. This is the isolation boundary
          for PRIVATE, single-owner corpora; nothing below can widen it.
        * ``visibility="shared"`` — a SHARED-intel corpus (e.g. CERT/ACI reports) where
          one document is visible to many orgs. Visible when the report is PUBLIC
          (``customer_tags`` empty/absent) OR explicitly allow-listed for this org
          (``customer_tags`` array contains ``org_id``). NOTE: this is FAIL-OPEN — a
          report with no tags is visible to EVERY tenant — so only bind it to corpora
          that are genuinely cross-tenant shareable.

        The optional SearchFilters are added on top in either mode (narrow only)."""
        from qdrant_client import models as qm

        if visibility == "shared":
            # (public OR tagged-for-this-org). Wrapped in its own Filter so it ANDs
            # cleanly with any source/date/doc musts appended below.
            must = [qm.Filter(should=[
                qm.FieldCondition(key="customer_tags", match=qm.MatchValue(value=org_id)),   # array contains org_id
                qm.IsEmptyCondition(is_empty=qm.PayloadField(key="customer_tags")),          # missing / null / []
            ])]
        else:
            # MANDATORY: org_id must match. The tenant-isolation boundary for private
            # corpora, enforced by Qdrant itself rather than post-filtering in Python.
            must = [qm.FieldCondition(key="org_id", match=qm.MatchValue(value=org_id))]
        if f:
            if f.source:
                must.append(qm.FieldCondition(key="source", match=qm.MatchValue(value=f.source)))
            if f.doc_ids:
                # MatchAny = "doc_id is one of these" (a SQL IN (...)).
                must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=list(f.doc_ids))))
            if f.date_from or f.date_to:
                # Recency window pushed down to the DB: drop points outside
                # [date_from, date_to] before ANN ranking (the time-decay fix).
                must.append(qm.FieldCondition(
                    key="published_at",
                    range=qm.DatetimeRange(gte=f.date_from or None, lte=f.date_to or None),
                ))
            for k, v in (f.extra or {}).items():
                # Arbitrary equality on nested payload, addressed as metadata.<key>.
                must.append(qm.FieldCondition(key=f"metadata.{k}", match=qm.MatchValue(value=v)))
        return qm.Filter(must=must)

    async def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        *,
        org_id: str,
        top_k: int,
        filters: SearchFilters | None = None,
        visibility: str = "tenant",
    ) -> list[Chunk]:
        # Fail closed without a tenant scope — required even in shared mode, since the
        # allow-list is matched against THIS org's id from the trusted context.
        if not org_id:
            raise ValueError("search requires org_id (tenant isolation)")
        # ANN search: Qdrant returns the nearest vectors to ``query_vector`` that also
        # satisfy ``query_filter`` — the filter always carries the tenant/visibility
        # clause, so results are scoped by construction.
        res = await self._client.query_points(
            collection_name=collection,
            query=list(query_vector),
            query_filter=self._filter(org_id, filters, visibility),
            limit=max(1, top_k),
            with_payload=True,                # we need the payload to rebuild Chunks
        )
        out: list[Chunk] = []
        # Rehydrate each hit's payload back into a Chunk (text + provenance + the
        # similarity ``score``) for the pipeline.
        for p in res.points:
            pl = p.payload or {}
            out.append(Chunk(
                id=pl.get("rid", str(p.id)), text=pl.get("text", ""), score=float(p.score or 0.0),
                org_id=pl.get("org_id", org_id), source=pl.get("source", ""),
                doc_id=pl.get("doc_id", ""), title=pl.get("title", ""),
                section=pl.get("section", ""), published_at=pl.get("published_at"),
                metadata=pl.get("metadata", {}) or {},
            ))
        return out

    async def delete_by_doc(self, collection: str, org_id: str, doc_id: str) -> int:
        from qdrant_client import models as qm

        # Delete every point for this doc — but ONLY within this org (both
        # conditions in ``must``), so deletion can't cross the tenant boundary.
        await self._client.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                qm.FieldCondition(key="org_id", match=qm.MatchValue(value=org_id)),
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
            ])),
        )
        return 0  # qdrant delete does not return a precise count here

    async def count(self, collection: str, org_id: str | None = None) -> int:
        from qdrant_client import models as qm

        # Count all points, or just one tenant's when org_id is given. ``exact=True``
        # asks Qdrant for a precise (not estimated) count.
        flt = None
        if org_id:
            flt = qm.Filter(must=[qm.FieldCondition(key="org_id", match=qm.MatchValue(value=org_id))])
        res = await self._client.count(collection_name=collection, count_filter=flt, exact=True)
        return int(res.count)

    async def aclose(self) -> None:
        await self._client.close()


__all__ = ["QdrantVectorStore"]
