"""Qdrant vector store (``retrieval_backend=qdrant``).

Same contract as the in-memory store, with the **mandatory org_id filter applied
at the database layer** (a ``must`` match), payload indexes for fast filtered
hybrid search, and datetime range filters that fix the "last year returns old
data" bug at the DB instead of in post-ranking. Lazy-imports ``qdrant-client``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from app.core.contracts import Chunk
from app.core.rag.vector_store import SearchFilters, VectorRecord

_NS = uuid.UUID("00000000-0000-0000-0000-0000000000aa")


def _point_id(rid: str) -> str:
    try:
        return str(uuid.UUID(rid))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(_NS, rid))


class QdrantVectorStore:
    backend = "qdrant"

    def __init__(self, url: str, api_key: str = "", timeout: float = 30.0) -> None:
        from qdrant_client import AsyncQdrantClient  # lazy

        self._client = AsyncQdrantClient(url=url, api_key=api_key or None, timeout=timeout)

    async def ensure_collection(self, collection: str, dim: int) -> None:
        from qdrant_client import models as qm

        exists = await self._client.collection_exists(collection)
        if not exists:
            await self._client.create_collection(
                collection_name=collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
        # Payload indexes (idempotent; ignore if already present).
        for field, schema in [
            ("org_id", qm.PayloadSchemaType.KEYWORD),
            ("source", qm.PayloadSchemaType.KEYWORD),
            ("doc_id", qm.PayloadSchemaType.KEYWORD),
            ("published_at", qm.PayloadSchemaType.DATETIME),
        ]:
            try:
                await self._client.create_payload_index(collection, field_name=field, field_schema=schema)
            except Exception:  # noqa: BLE001 - already exists
                pass

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> int:
        from qdrant_client import models as qm

        points = []
        for r in records:
            if not r.org_id:
                raise ValueError("VectorRecord.org_id is required (tenant isolation)")
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

    def _filter(self, org_id: str, f: SearchFilters | None):
        from qdrant_client import models as qm

        must = [qm.FieldCondition(key="org_id", match=qm.MatchValue(value=org_id))]
        if f:
            if f.source:
                must.append(qm.FieldCondition(key="source", match=qm.MatchValue(value=f.source)))
            if f.doc_ids:
                must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=list(f.doc_ids))))
            if f.date_from or f.date_to:
                must.append(qm.FieldCondition(
                    key="published_at",
                    range=qm.DatetimeRange(gte=f.date_from or None, lte=f.date_to or None),
                ))
            for k, v in (f.extra or {}).items():
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
    ) -> list[Chunk]:
        if not org_id:
            raise ValueError("search requires org_id (tenant isolation)")
        res = await self._client.query_points(
            collection_name=collection,
            query=list(query_vector),
            query_filter=self._filter(org_id, filters),
            limit=max(1, top_k),
            with_payload=True,
        )
        out: list[Chunk] = []
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

        flt = None
        if org_id:
            flt = qm.Filter(must=[qm.FieldCondition(key="org_id", match=qm.MatchValue(value=org_id))])
        res = await self._client.count(collection_name=collection, count_filter=flt, exact=True)
        return int(res.count)

    async def aclose(self) -> None:
        await self._client.close()


__all__ = ["QdrantVectorStore"]
