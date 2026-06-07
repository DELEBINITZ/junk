"""The shared, corpus-agnostic retrieval pipeline.

THIS IS THE "R" IN RAG, assembled end to end. Every other file in this package is
one stage; this file wires them into the sequence a query actually flows through:

    embed query -> (auto) temporal filter -> org-scoped vector search -> recency
    re-weight -> rerank -> top-k chunks

Stage by stage, and where each lives:
  * embed query            (embeddings.py)  — turn the question into a vector
  * (auto) temporal filter (filters.py)     — parse "last year" into a date window
  * org-scoped vector search (vector_store / qdrant_backend) — nearest chunks,
    HARD-scoped to the caller's org_id (tenant isolation) — over-FETCHED on purpose
  * recency re-weight      (here)           — softly down-weight stale chunks
  * rerank                 (reranker.py)    — cross-encoder precision over the
    over-fetched candidates, then keep top_k

ONE pipeline serves every corpus; a module binds a :class:`CollectionRetriever` to
its collection — it never re-implements retrieval. This is how "not just reports"
works: add a collection, bind a retriever, done.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from app.config import Settings
from app.core.contracts import Chunk, ToolContext
from app.core.rag.embeddings import Embedder, build_embedder
from app.core.rag.filters import extract_time_filters
from app.core.rag.reranker import Reranker, build_reranker
from app.core.rag.vector_store import SearchFilters, VectorStore


def _age_days(published_at: str | None) -> float | None:
    # How old is this chunk, in days? Parse the timestamp (trying a few common
    # formats) and diff against now. Returns None when there's no/unparseable date,
    # which the recency step treats as "don't adjust". Drives time-decay scoring.
    if not published_at:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(published_at.replace("Z", "+0000"), fmt)
            if dt.tzinfo is None:                # assume UTC for naive timestamps
                dt = dt.replace(tzinfo=UTC)
            return max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400.0)
        except ValueError:
            continue
    return None


class RetrievalPipeline:
    # Holds the three swappable stages (embedder, vector store, reranker) plus
    # settings. Built once at boot (build_rag) and shared by every retriever.
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

    # NOTE: this pipeline is READ-ONLY. Documents are written to the vector store by
    # an external ingestion cron (embed + upsert directly to the collection); the app
    # only retrieves. There is intentionally no index()/upsert path here.

    def _apply_recency(self, chunks: list[Chunk]) -> list[Chunk]:
        """RECENCY RE-WEIGHTING (time-decay): softly nudge fresher chunks up the
        ranking without hard-dropping old ones (that's the date FILTER's job). Uses
        exponential decay with a half-life: a chunk one half-life old gets factor
        0.5, two half-lives 0.25, and so on. We keep a floor of 0.7 (only the 0.3
        portion decays) so age refines but never dominates the semantic score."""
        hl = self.settings.recency_half_life_days
        if hl <= 0:                      # 0/negative disables recency entirely
            return chunks
        rescored = []
        for c in chunks:
            age = _age_days(c.published_at)
            if age is None:              # undated chunk -> leave its score untouched
                rescored.append(c)
                continue
            factor = 0.5 ** (age / hl)   # exponential time-decay, in [0, 1]
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
        visibility: str = "tenant",
    ) -> list[Chunk]:
        """The READ path — the full retrieve sequence for one query."""
        tracer = getattr(ctx.deps, "tracer", None)
        sf = filters or SearchFilters()
        if apply_time_filters and not (sf.date_from or sf.date_to):
            df, dt = extract_time_filters(query)
            if df or dt:
                sf = sf.model_copy(update={"date_from": df, "date_to": dt})

        qv = await self.embedder.embed_query(query)
        over_fetch = max(self.settings.retrieval_top_k, top_k or 0)
        chunks = await self.store.search(
            collection, qv, org_id=ctx.org_id, top_k=over_fetch, filters=sf, visibility=visibility
        )
        chunks = self._apply_recency(chunks)
        final_k = top_k or self.settings.rerank_top_k
        if self.settings.rerank_enabled and chunks:
            chunks = await self.reranker.rerank(query, chunks, final_k)
        else:
            chunks = chunks[:final_k]

        if tracer:
            tracer.event("rag.retrieve", collection=collection, org_id=ctx.org_id,
                         query_len=len(query), results=len(chunks), top_k=final_k)
        return chunks

    async def aclose(self) -> None:
        # Release every stage that holds resources (HTTP clients, DB connections)
        # on shutdown. Guarded with getattr so a stage without aclose is fine.
        for c in (self.embedder, self.store, self.reranker):
            close = getattr(c, "aclose", None)
            if close:
                await close()


class CollectionRetriever:
    """Binds the shared pipeline to one collection. Implements ``contracts.Retriever``.

    Modules declare this statically in their manifest (``pipeline=None``); at call
    time it uses the live pipeline from ``ctx.deps.rag``. Pass an explicit
    ``pipeline`` for programmatic use."""

    def __init__(self, id: str, collection: str, pipeline: RetrievalPipeline | None = None,
                 source: str = "", visibility: str = "tenant") -> None:
        self.id = id
        self.collection = collection
        self.pipeline = pipeline
        self.source = source
        # "tenant" = hard org isolation (default, private corpora). "shared" = the
        # public-OR-customer_tags allow-list, for cross-tenant shared corpora.
        self.visibility = visibility

    async def retrieve(
        self, query: str, filters: Mapping[str, Any], ctx: ToolContext
    ) -> list[Chunk]:
        # Satisfies contracts.Retriever. Grab the live pipeline (from ctx.deps when
        # bound statically in a manifest), pin it to THIS retriever's collection,
        # and delegate. ctx carries the org_id that scopes the search.
        pipeline = self.pipeline or ctx.deps.rag
        sf = SearchFilters(**dict(filters)) if filters else None
        return await pipeline.retrieve(query, collection=self.collection, ctx=ctx,
                                       filters=sf, visibility=self.visibility)


def build_vector_store(settings: Settings) -> VectorStore:
    # Qdrant is the only vector backend (org_id-scoped). No in-memory fallback.
    from app.core.rag.qdrant_backend import QdrantVectorStore

    return QdrantVectorStore(settings.qdrant_url, settings.qdrant_api_key,
                             shared_untagged_public=settings.shared_untagged_public)


def build_rag(settings: Settings) -> RetrievalPipeline:
    # Compose the whole pipeline from config — each stage chosen by its own factory.
    # This is the single place the RAG stack is assembled (called once at boot).
    return RetrievalPipeline(
        embedder=build_embedder(settings),
        store=build_vector_store(settings),
        reranker=build_reranker(settings),
        settings=settings,
    )


__all__ = [
    "RetrievalPipeline",
    "CollectionRetriever",
    "build_vector_store",
    "build_rag",
]
