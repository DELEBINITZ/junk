"""TEI reranker client (Qwen3-Reranker) — the biggest single retrieval-quality
win (plan §7.2). Cross-encodes (query, passage) pairs and returns scored order.

Active when RERANK_ENABLED=true and a TEI rerank endpoint is reachable. httpx is
already a dependency; the call only happens when enabled.
"""

from __future__ import annotations

import httpx

from app.config import settings


class TEIReranker:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 30.0):
        self.base_url = (base_url or settings.tei_rerank_base_url).rstrip("/")
        self.model = model or settings.tei_rerank_model
        self.timeout = timeout

    def rerank(self, query: str, passages: list[str], top_k: int) -> list[tuple[int, float]]:
        """Return [(original_index, score), ...] for the top_k passages."""

        if not passages:
            return []
        response = httpx.post(
            f"{self.base_url}/rerank",
            json={"query": query, "texts": passages},
            timeout=self.timeout,
        )
        response.raise_for_status()
        ranked = [(int(item["index"]), float(item["score"])) for item in response.json()]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]
