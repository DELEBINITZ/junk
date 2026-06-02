"""The shared retrieval pipeline — what makes "RAG over reports" generalize to
"RAG over any corpus".

Backend is config-switched (RETRIEVAL_BACKEND): "memory" runs the existing
in-memory ranker over an RBAC + org-filtered chunk set (default; no deps);
"qdrant" runs hybrid filtered search with a MANDATORY org_id filter at the vector
layer. The (query, ctx, filters) -> chunks interface is identical either way, so
a module's Retriever never changes. `collection` is the per-corpus seam. See
plan §7 and §8.2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.core.contracts import ToolContext
from app.core.rag.qdrant_backend import QdrantRetriever
from app.rag.retrieval import search_chunks
from app.rbac.permissions import queryable_documents


def _default_embedder():
    if settings.embedding_provider.lower() == "tei":
        from app.core.rag.embeddings_tei import TEIEmbeddingProvider

        return TEIEmbeddingProvider()
    from app.rag.embeddings import create_embedding_provider

    return create_embedding_provider()


@dataclass
class RetrievalPipeline:
    collection: str = "reports_kb"
    qdrant: Any = None    # injectable QdrantRetriever (tests / DI)
    embedder: Any = None  # injectable embedding provider (tests / DI)

    def retrieve(
        self,
        query: str,
        ctx: ToolContext,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        backend = os.getenv("RETRIEVAL_BACKEND", settings.retrieval_backend).lower()
        if backend == "qdrant":
            return self._qdrant_retrieve(query, ctx, top_k, filters or {})
        return self._memory_retrieve(query, ctx, top_k, filters or {})

    def _memory_retrieve(self, query: str, ctx: ToolContext, top_k: int, filters: dict) -> list[dict]:
        store = ctx.store
        # Org + RBAC filtering happens BEFORE ranking (cross-org chunks are never
        # scored). On Qdrant this is the mandatory org_id payload filter below.
        documents = queryable_documents(ctx.user, store)
        if filters.get("tags"):
            wanted = set(filters["tags"])
            documents = [d for d in documents if wanted.intersection(d.tags)]
        document_map = {d.id: d for d in documents}
        chunks = store.chunks_for_documents(document_map.keys(), organization_id=ctx.org_id)
        hits = search_chunks(query, chunks, document_map, store.embedder, top_k=top_k)
        return [
            {
                "contract_id": hit.document.contract_id,
                "title": hit.document.title,
                "section_number": hit.chunk.metadata.get("section_number"),
                "section_title": hit.chunk.metadata.get("section_title"),
                "snippet": hit.chunk.text[:600],
                "score": round(hit.score, 4),
                "citation": f"[{hit.document.contract_id}, Section {hit.chunk.metadata.get('section_number')}]",
            }
            for hit in hits
        ]

    def _qdrant_retrieve(self, query: str, ctx: ToolContext, top_k: int, filters: dict) -> list[dict]:
        retriever = self.qdrant or QdrantRetriever(collection=self.collection)
        embedder = self.embedder or getattr(ctx.store, "embedder", None) or _default_embedder()
        vector = embedder.embed(query)
        # org_id from the trusted context — the mandatory tenant filter.
        # Over-fetch then rerank when enabled (cross-encoder precision).
        fetch_k = top_k * 4 if settings.rerank_enabled else top_k
        hits = retriever.search(ctx.org_id, vector, top_k=fetch_k, filters=filters)
        if settings.rerank_enabled and len(hits) > 1:
            hits = self._rerank(query, hits, top_k)
        return hits[:top_k]

    def _rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        from app.core.rag.reranker_tei import TEIReranker

        try:
            ranked = TEIReranker().rerank(query, [h.get("snippet", "") for h in hits], top_k)
        except Exception:
            return hits  # fail open to vector order
        reordered = []
        for index, score in ranked:
            hit = dict(hits[index])
            hit["rerank_score"] = round(score, 4)
            reordered.append(hit)
        return reordered
