"""Cross-encoder reranking — the single biggest retrieval-quality win.

THE OVER-FETCH-THEN-RERANK PATTERN (why two retrieval stages, not one): vector
search is FAST but coarse — it compares the query and each chunk as separate
pre-computed embeddings, which can rank a loosely-related chunk above the truly
best one. A CROSS-ENCODER is slower but far more accurate: it feeds the (query,
chunk) PAIR through a model TOGETHER, so the model can directly judge "does this
chunk answer this query?". You can't run a cross-encoder over the whole corpus
(too slow), so the pipeline does both: over-fetch a generous candidate set with
cheap vector search, then rerank just those candidates with the cross-encoder and
keep the top_k. Best of both — vector recall, cross-encoder precision.

This file is the swappable reranker stage. Default ``none`` keeps vector order
(no rerank). ``tei`` calls a Text-Embeddings-Inference reranker (a real
cross-encoder, Qwen3-Reranker in prod). A lexical reranker is provided for
infra-free demonstration of the over-fetch-then-rerank flow.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from app.config import Settings
from app.core.contracts import Chunk

_WORD = re.compile(r"[a-z0-9]+")     # word tokenizer for the lexical reranker


class Reranker(Protocol):
    # The reranker interface (Protocol). Takes the query + the over-fetched
    # candidates and returns the reordered top_k. All three impls below satisfy it.
    async def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[Chunk]: ...
    async def aclose(self) -> None: ...


class NoopReranker:
    """No reranking: trust the vector store's order and just truncate to top_k.
    Selected when reranking is disabled — the cheapest path."""

    provider = "none"

    async def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[Chunk]:
        return list(chunks)[:top_k]

    async def aclose(self) -> None:
        return None


class LexicalReranker:
    """Token-overlap reranker — deterministic, no infra. Useful to demonstrate
    over-fetch + rerank without a model server. NOT a true cross-encoder: it just
    blends the vector score with query/chunk word overlap as a stand-in."""

    provider = "lexical"

    async def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[Chunk]:
        q = set(_WORD.findall(query.lower()))
        scored = []
        for c in chunks:
            t = set(_WORD.findall(c.text.lower()))
            overlap = len(q & t) / (len(q) or 1)      # fraction of query words present in the chunk
            # Blend 50/50: keep some of the original semantic score, add the lexical
            # signal — so an exact keyword hit can be promoted above a fuzzy match.
            scored.append((0.5 * c.score + 0.5 * overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _s, c in scored[:top_k]]        # reordered, then cut to top_k

    async def aclose(self) -> None:
        return None


class TEIReranker:
    """The real cross-encoder path: POST the query + candidate texts to a TEI
    ``/rerank`` model server (Qwen3-Reranker in prod), which scores each (query,
    chunk) pair jointly and returns relevance scores to reorder by."""

    provider = "tei"

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = None

    def _http(self):
        # Lazy, reused async HTTP client (no socket opened at construction).
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[Chunk]:
        from app.core.errors import UpstreamError

        if not chunks:
            return []
        try:
            # Send the query alongside every candidate's text; the server returns a
            # relevance score per candidate (by its index in the list we sent).
            r = await self._http().post(
                f"{self._base_url}/rerank",
                json={"query": query, "texts": [c.text for c in chunks]},
            )
            r.raise_for_status()
            results = r.json()  # [{"index": i, "score": s}, ...]
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(f"TEI rerank failed: {exc}") from exc
        # Reorder by the cross-encoder's score (best first) and keep top_k. Each
        # surviving chunk's score is overwritten with the reranker's score so
        # downstream sees the better signal; model_copy avoids mutating the input.
        order = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        out: list[Chunk] = []
        for item in order[:top_k]:
            c = chunks[item["index"]]
            out.append(c.model_copy(update={"score": float(item.get("score", c.score))}))
        return out

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def build_reranker(settings: Settings) -> Reranker:
    # Factory: reranking off -> Noop; "tei" -> real cross-encoder server; otherwise
    # the infra-free lexical stand-in. The pipeline only ever sees the Reranker
    # interface, so swapping these changes quality, not call sites.
    if not settings.rerank_enabled:
        return NoopReranker()
    if settings.rerank_provider == "tei":
        return TEIReranker(settings.tei_rerank_url)
    return LexicalReranker()


__all__ = ["Reranker", "NoopReranker", "LexicalReranker", "TEIReranker", "build_reranker"]
