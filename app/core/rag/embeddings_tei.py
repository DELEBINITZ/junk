"""TEI embedding client (Qwen3-Embedding 8B) — plan §7/§14.

Talks to a Text-Embeddings-Inference server over HTTP. Selected when
EMBEDDING_PROVIDER=tei; the default remains the in-repo hash provider so local
dev/tests need no model server. Mirrors the EmbeddingProvider surface
(`embed`, `dimensions`).
"""

from __future__ import annotations

import httpx

from app.config import settings


class TEIEmbeddingProvider:
    def __init__(self, base_url: str | None = None, model: str | None = None, dimensions: int | None = None):
        self.base_url = (base_url or settings.tei_base_url).rstrip("/")
        self.model = model or settings.tei_model
        self.dimensions = dimensions or settings.embedding_dimensions

    def embed(self, text: str) -> list[float]:
        response = httpx.post(f"{self.base_url}/embed", json={"inputs": text}, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        # TEI returns a list of vectors (one per input).
        vector = data[0] if data and isinstance(data[0], list) else data
        return list(vector)
