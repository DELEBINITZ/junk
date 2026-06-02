"""Embedders.

Default ``deterministic`` uses a *stable* hashed bag-of-words embedding (blake2,
not Python's randomized ``hash()``), so cosine similarity is meaningful and
reproducible across processes/runs with zero infra. ``tei`` calls a
Text-Embeddings-Inference server (Qwen3-Embedding in prod); ``openai`` calls an
OpenAI-compatible embeddings endpoint.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.config import Settings

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def aclose(self) -> None: ...


class DeterministicEmbedder:
    provider = "deterministic"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _tokens(text)
        for tok in toks:
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0
        # add bigrams for a little word-order signal
        for a, b in zip(toks, toks[1:], strict=False):
            h = int.from_bytes(hashlib.blake2b(f"{a}_{b}".encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 0.5
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t).tolist() for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text).tolist()

    async def aclose(self) -> None:
        return None


class TEIEmbedder:
    """Hugging Face Text-Embeddings-Inference (``/embed``)."""

    provider = "tei"

    def __init__(self, base_url: str, dim: int, timeout: float = 30.0) -> None:
        self.dim = dim
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from app.core.errors import UpstreamError

        try:
            r = await self._http().post(f"{self._base_url}/embed", json={"inputs": list(texts)})
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(f"TEI embed failed: {exc}") from exc

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class OpenAIEmbedder:
    provider = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, dim: int, timeout: float = 30.0) -> None:
        self.dim = dim
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = None

    def _http(self):
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
            return [d["embedding"] for d in r.json()["data"]]
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(f"OpenAI embed failed: {exc}") from exc

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def build_embedder(settings: Settings) -> Embedder:
    p = settings.embedding_provider
    if p == "deterministic":
        return DeterministicEmbedder(dim=settings.embedding_dim)
    if p == "tei":
        return TEIEmbedder(settings.tei_embed_url, dim=settings.embedding_dim)
    if p == "openai":
        return OpenAIEmbedder(
            settings.openai_base_url, settings.openai_api_key,
            settings.embedding_model, dim=settings.embedding_dim,
        )
    from app.core.errors import ConfigError

    raise ConfigError(f"unknown embedding_provider: {p}")


__all__ = [
    "Embedder",
    "DeterministicEmbedder",
    "TEIEmbedder",
    "OpenAIEmbedder",
    "build_embedder",
]
