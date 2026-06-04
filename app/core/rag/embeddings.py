"""Embedders.

WHAT AN EMBEDDING IS (the core of RAG retrieval): an embedding turns a piece of
text into a fixed-length vector of floats — a point in a high-dimensional space —
positioned so that texts with similar MEANING land near each other. We embed every
chunk at index time and embed the question at query time, then find the chunks
whose vectors are closest to the question's. "Closest" is measured by COSINE
SIMILARITY: the cosine of the angle between two vectors, which is 1.0 when they
point the same way and 0 when unrelated. (Cosine, not raw distance, is why all the
embedders here L2-normalize their vectors.) ``dim`` is the vector length / number
of dimensions; the query and the chunks MUST share the same ``dim`` to be compared.

This file is the swappable embedding backend. Default ``deterministic`` uses a
*stable* hashed bag-of-words embedding (blake2, not Python's randomized ``hash()``
which differs per process), so cosine similarity is meaningful and reproducible
across processes/runs with zero infra. ``tei`` calls a Text-Embeddings-Inference
server (Qwen3-Embedding in prod); ``openai`` calls an OpenAI-compatible embeddings
endpoint. All three satisfy the same ``Embedder`` protocol so the pipeline doesn't
care which is wired.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.config import Settings


class Embedder(Protocol):
    # The contract every embedder implements (a Protocol = duck-typed interface,
    # no base class to inherit). ``dim`` is the output vector length.
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...      # batch: documents at index time

    async def embed_query(self, text: str) -> list[float]: ...                 # single: the question at query time

    async def aclose(self) -> None: ...                                        # release any HTTP client


class TEIEmbedder:
    """Hugging Face Text-Embeddings-Inference (``/embed``).

    Calls a self-hosted embedding model server over HTTP (Qwen3-Embedding in prod).
    This is the real-quality path: a trained model produces genuinely semantic
    vectors, at the cost of running a GPU service."""

    provider = "tei"

    def __init__(self, base_url: str, dim: int, timeout: float = 30.0, query_instruction: str = "") -> None:
        self.dim = dim
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = None
        # Instruct-embedder query prefix (Qwen3). Applied to embed_query ONLY; "" = off.
        self._query_instruction = query_instruction

    def _http(self):
        # Lazily create (and reuse) the async HTTP client on first use, so merely
        # constructing the embedder opens no sockets.
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from app.core.errors import UpstreamError

        # Wrap any network/HTTP failure as a domain UpstreamError so callers see a
        # consistent error type rather than raw httpx exceptions.
        try:
            r = await self._http().post(f"{self._base_url}/embed", json={"inputs": list(texts)})
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(f"TEI embed failed: {exc}") from exc

    async def embed_query(self, text: str) -> list[float]:
        # Embed one text by reusing the batch path and taking the only result. Prepend
        # the instruction prefix (Qwen3 query side) when configured; docs stay raw.
        q = f"{self._query_instruction}{text}" if self._query_instruction else text
        return (await self.embed([q]))[0]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class OpenAIEmbedder:
    """OpenAI-compatible ``/embeddings`` endpoint (also fits Azure/other vendors
    speaking the same API). ``model`` names the hosted embedding model."""

    provider = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, dim: int, timeout: float = 30.0, query_instruction: str = "") -> None:
        self.dim = dim
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = None
        self._query_instruction = query_instruction

    def _http(self):
        # Lazy client, with the bearer token set once as a default header.
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=self._timeout, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        return self._client

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from app.core.errors import UpstreamError

        try:
            r = await self._http().post(
                f"{self._base_url}/embeddings", json={"model": self._model, "input": list(texts)}
            )
            r.raise_for_status()
            # OpenAI returns {"data": [{"embedding": [...]}, ...]} — pull the vectors.
            return [d["embedding"] for d in r.json()["data"]]
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(f"OpenAI embed failed: {exc}") from exc

    async def embed_query(self, text: str) -> list[float]:
        q = f"{self._query_instruction}{text}" if self._query_instruction else text
        return (await self.embed([q]))[0]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def build_embedder(settings: Settings) -> Embedder:
    # Factory: pick the embedder from config. ``embedding_dim`` must match what the
    # chosen model actually outputs (e.g. Qwen3 supports Matryoshka embeddings —
    # one model trained so you can truncate the vector to a shorter ``dim`` and
    # still get usable similarity, trading a little accuracy for speed/storage).
    p = settings.embedding_provider
    qi = settings.embedding_query_instruction
    if p == "tei":
        return TEIEmbedder(settings.tei_embed_url, dim=settings.embedding_dim, query_instruction=qi)
    if p == "openai":
        return OpenAIEmbedder(
            settings.openai_base_url, settings.openai_api_key,
            settings.embedding_model, dim=settings.embedding_dim, query_instruction=qi,
        )
    from app.core.errors import ConfigError

    raise ConfigError(f"unknown embedding_provider: {p}")


__all__ = [
    "Embedder",
    "TEIEmbedder",
    "OpenAIEmbedder",
    "build_embedder",
]
