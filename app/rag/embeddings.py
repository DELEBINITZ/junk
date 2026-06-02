"""Embedding providers used by retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from app.config import settings

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


class EmbeddingProvider:
    dimensions: int

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class HashingEmbeddingProvider(EmbeddingProvider):
    """Small deterministic embedding substitute for local tests and demos.

    Production can replace this with BGE through the same `embed(text)` method.
    The default dimension is 384 to match `BAAI/bge-small-en-v1.5`.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class BGEEmbeddingProvider(EmbeddingProvider):
    """Optional BGE provider backed by sentence-transformers.

    This is intentionally imported lazily so the default demo can run without
    downloading model weights. Set `EMBEDDING_PROVIDER=bge` after installing the
    optional `prod` dependencies.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        sentence_transformers = _import_sentence_transformers()
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.dimensions = int(self.model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in embedding]


def create_embedding_provider() -> EmbeddingProvider:
    """Create the configured embedding provider."""

    if settings.embedding_provider.lower() == "bge":
        return BGEEmbeddingProvider(settings.embedding_model)
    return HashingEmbeddingProvider(settings.embedding_dimensions)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine score for normalized embeddings."""

    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _import_sentence_transformers() -> Any:
    """Import sentence-transformers lazily so the default demo has no model dependency."""

    try:
        import sentence_transformers
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for EMBEDDING_PROVIDER=bge. "
            "Install the project with the 'prod' extra or set EMBEDDING_PROVIDER=hash."
        ) from exc
    return sentence_transformers
