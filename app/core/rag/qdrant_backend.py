"""Qdrant retrieval backend (plan §7).

Selected when RETRIEVAL_BACKEND=qdrant. The `organization_id` payload filter is
MANDATORY on every query — org isolation at the vector layer, not a tool arg the
model can omit (plan §8.2). `qdrant-client` is lazy-imported so the default
in-memory path needs no driver.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings


logger = logging.getLogger(__name__)


class QdrantRetriever:
    def __init__(self, url: str | None = None, collection: str | None = None, client: Any = None):
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self._client = client  # injectable for tests

    def _qdrant(self):
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient  # lazy

        self._client = QdrantClient(url=self.url)
        return self._client

    # ---- ingestion side (create collection + upsert embedded points) ------------
    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection (cosine) + payload indexes if absent. Idempotent."""

        from qdrant_client import models  # lazy

        client = self._qdrant()
        names = {c.name for c in client.get_collections().collections}
        if self.collection not in names:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        # Payload indexes for filtered retrieval. organization_id is the tenant key.
        for field, schema in (
            ("organization_id", "keyword"), ("contract_id", "keyword"),
            ("tags", "keyword"), ("doc_type", "keyword"), ("year", "integer"),
        ):
            try:
                client.create_payload_index(self.collection, field_name=field, field_schema=schema)
            except Exception:
                pass  # index already exists

    def upsert(self, points: list[dict]) -> int:
        """Upsert points: each {id, vector, payload}. payload MUST carry
        organization_id (the tenant tag every query filters on)."""

        from qdrant_client import models  # lazy

        client = self._qdrant()
        client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points
            ],
        )
        return len(points)

    def search(
        self,
        organization_id: str,
        query_vector: list[float],
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        from qdrant_client import models  # lazy

        must = [
            models.FieldCondition(
                key="organization_id",
                match=models.MatchValue(value=organization_id),
            )
        ]
        if filters and filters.get("tags"):
            must.append(
                models.FieldCondition(key="tags", match=models.MatchAny(any=list(filters["tags"])))
            )
        result = self._qdrant().query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=models.Filter(must=must),
            limit=top_k,
            with_payload=True,
        )
        hits = []
        for point in result.points:
            payload = point.payload or {}
            contract_id = payload.get("contract_id")
            section = payload.get("section_number")
            hits.append(
                {
                    "contract_id": contract_id,
                    "title": payload.get("title"),
                    "section_number": section,
                    "section_title": payload.get("section_title"),
                    "snippet": (payload.get("text") or "")[:600],
                    "score": round(float(point.score), 4),
                    "citation": f"[{contract_id}, Section {section}]",
                }
            )
        return hits
