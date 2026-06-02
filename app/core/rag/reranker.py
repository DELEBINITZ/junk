"""Cross-encoder reranking — the single biggest retrieval-quality win.

Default ``none`` keeps vector order. ``tei`` calls a Text-Embeddings-Inference
reranker (Qwen3-Reranker in prod). A lexical reranker is provided for infra-free
demonstration of over-fetch-then-rerank.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from app.config import Settings
from app.core.contracts import Chunk

_WORD = re.compile(r"[a-z0-9]+")


class Reranker(Protocol):
    async def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[Chunk]: ...
    async def aclose(self) -> None: ...


class NoopReranker:
    provider = "none"

    async def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[Chunk]:
        return list(chunks)[:top_k]

    async def aclose(self) -> None:
        return None


class LexicalReranker:
    """Token-overlap reranker — deterministic, no infra. Useful to demonstrate
    over-fetch + rerank without a model server."""

    provider = "lexical"

    async def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[Chunk]:
        q = set(_WORD.findall(query.lower()))
        scored = []
        for c in chunks:
            t = set(_WORD.findall(c.text.lower()))
            overlap = len(q & t) / (len(q) or 1)
            scored.append((0.5 * c.score + 0.5 * overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _s, c in scored[:top_k]]

    async def aclose(self) -> None:
        return None


class TEIReranker:
    provider = "tei"

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[Chunk]:
        from app.core.errors import UpstreamError

        if not chunks:
            return []
        try:
            r = await self._http().post(
                f"{self._base_url}/rerank",
                json={"query": query, "texts": [c.text for c in chunks]},
            )
            r.raise_for_status()
            results = r.json()  # [{"index": i, "score": s}, ...]
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(f"TEI rerank failed: {exc}") from exc
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
    if not settings.rerank_enabled:
        return NoopReranker()
    if settings.rerank_provider == "tei":
        return TEIReranker(settings.tei_rerank_url)
    return LexicalReranker()


__all__ = ["Reranker", "NoopReranker", "LexicalReranker", "TEIReranker", "build_reranker"]
