"""The shared, corpus-agnostic retrieval pipeline.

embed query -> (auto) temporal filter -> org-scoped vector search -> recency
re-weight -> rerank -> top-k chunks. ONE pipeline serves every corpus; a module
binds a :class:`CollectionRetriever` to its collection — it never re-implements
retrieval. This is how "not just reports" works: add a collection, bind a
retriever, done.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings
from app.core.contracts import Chunk, ToolContext
from app.core.rag.embeddings import Embedder, build_embedder
from app.core.rag.filters import extract_time_filters
from app.core.rag.reranker import Reranker, build_reranker
from app.core.rag.vector_store import InMemoryVectorStore, SearchFilters, VectorRecord, VectorStore


class IndexItem(BaseModel):
    id: str
    text: str
    source: str = ""
    doc_id: str = ""
    title: str = ""
    section: str = ""
    published_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _age_days(published_at: str | None) -> float | None:
    if not published_at:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(published_at.replace("Z", "+0000"), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400.0)
        except ValueError:
            continue
    return None


class RetrievalPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.settings = settings

    async def ensure_collection(self, collection: str) -> None:
        await self.store.ensure_collection(collection, self.embedder.dim)

    async def index(self, collection: str, org_id: str, items: Sequence[IndexItem]) -> int:
        if not org_id:
            raise ValueError("index requires org_id (tenant isolation)")
        await self.ensure_collection(collection)
        vectors = await self.embedder.embed([it.text for it in items])
        records = [
            VectorRecord(
                id=it.id, org_id=org_id, vector=vectors[i], text=it.text, source=it.source,
                doc_id=it.doc_id, title=it.title, section=it.section,
                published_at=it.published_at, metadata=it.metadata,
            )
            for i, it in enumerate(items)
        ]
        return await self.store.upsert(collection, records)

    def _apply_recency(self, chunks: list[Chunk]) -> list[Chunk]:
        hl = self.settings.recency_half_life_days
        if hl <= 0:
            return chunks
        rescored = []
        for c in chunks:
            age = _age_days(c.published_at)
            if age is None:
                rescored.append(c)
                continue
            factor = 0.5 ** (age / hl)
            rescored.append(c.model_copy(update={"score": c.score * (0.7 + 0.3 * factor)}))
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored

    async def retrieve(
        self,
        query: str,
        *,
        collection: str,
        ctx: ToolContext,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
        apply_time_filters: bool = True,
    ) -> list[Chunk]:
        sf = filters or SearchFilters()
        if apply_time_filters and not (sf.date_from or sf.date_to):
            df, dt = extract_time_filters(query)
            if df or dt:
                sf = sf.model_copy(update={"date_from": df, "date_to": dt})

        qv = await self.embedder.embed_query(query)
        over_fetch = max(self.settings.retrieval_top_k, top_k or 0)
        chunks = await self.store.search(
            collection, qv, org_id=ctx.org_id, top_k=over_fetch, filters=sf
        )
        chunks = self._apply_recency(chunks)
        final_k = top_k or self.settings.rerank_top_k
        if self.settings.rerank_enabled and chunks:
            chunks = await self.reranker.rerank(query, chunks, final_k)
        else:
            chunks = chunks[:final_k]
        return chunks

    async def aclose(self) -> None:
        for c in (self.embedder, self.store, self.reranker):
            close = getattr(c, "aclose", None)
            if close:
                await close()


class CollectionRetriever:
    """Binds the shared pipeline to one collection. Implements ``contracts.Retriever``.

    Modules declare this statically in their manifest (``pipeline=None``); at call
    time it uses the live pipeline from ``ctx.deps.rag``. Pass an explicit
    ``pipeline`` for programmatic use."""

    def __init__(self, id: str, collection: str, pipeline: RetrievalPipeline | None = None, source: str = "") -> None:
        self.id = id
        self.collection = collection
        self.pipeline = pipeline
        self.source = source

    async def retrieve(
        self, query: str, filters: Mapping[str, Any], ctx: ToolContext
    ) -> list[Chunk]:
        pipeline = self.pipeline or ctx.deps.rag
        sf = SearchFilters(**dict(filters)) if filters else None
        return await pipeline.retrieve(query, collection=self.collection, ctx=ctx, filters=sf)


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.retrieval_backend == "qdrant":
        from app.core.rag.qdrant_backend import QdrantVectorStore

        return QdrantVectorStore(settings.qdrant_url, settings.qdrant_api_key)
    return InMemoryVectorStore()


def build_rag(settings: Settings) -> RetrievalPipeline:
    return RetrievalPipeline(
        embedder=build_embedder(settings),
        store=build_vector_store(settings),
        reranker=build_reranker(settings),
        settings=settings,
    )


__all__ = [
    "IndexItem",
    "RetrievalPipeline",
    "CollectionRetriever",
    "build_vector_store",
    "build_rag",
]
