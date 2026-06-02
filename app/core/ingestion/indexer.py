"""Production ingestion: embed documents and upsert them into Qdrant for RAG.

This is the write side of the vector path (the read side is RetrievalPipeline's
qdrant backend). Text is chunked, embedded with the configured provider (TEI in
prod), and upserted with `organization_id` stamped on every point — the mandatory
tenant tag the retriever filters on. The same embedder is used for indexing and
querying so vectors are comparable. See plan §7.1.

Wire to a Temporal worker or call from the /ingest API for event-driven ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from app.config import settings
from app.core.ingestion.connectors import IngestStats
from app.core.rag.pipeline import _default_embedder
from app.core.rag.qdrant_backend import QdrantRetriever


def _window_chunks(text: str, size: int = 350, overlap: int = 50) -> list[str]:
    """Generic word-window chunker for arbitrary production text (the section
    chunker is contract-specific). ~size words per chunk with overlap."""

    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        start += step
    return chunks


@dataclass
class QdrantIngestionService:
    collection: str | None = None
    embedder: object = None
    retriever: QdrantRetriever | None = None

    def __post_init__(self):
        self.collection = self.collection or settings.qdrant_collection
        self.embedder = self.embedder or _default_embedder()
        self.retriever = self.retriever or QdrantRetriever(collection=self.collection)

    def index_documents(self, organization_id: str, documents: list[dict]) -> IngestStats:
        """documents: [{contract_id, title, text, tags?, doc_type?, metadata?}].
        org comes from the caller's trusted context, never from the payload."""

        self.retriever.ensure_collection(self.embedder.dimensions)
        points: list[dict] = []
        for document in documents:
            contract_id = str(document.get("contract_id") or uuid4())
            title = document.get("title", contract_id)
            tags = list(document.get("tags") or [])
            base_payload = {
                "organization_id": organization_id,  # MANDATORY tenant tag
                "contract_id": contract_id,
                "title": title,
                "tags": tags,
                "doc_type": document.get("doc_type"),
                **(document.get("metadata") or {}),
            }
            for index, chunk_text in enumerate(_window_chunks(str(document.get("text", "")))):
                vector = self.embedder.embed(chunk_text)
                points.append(
                    {
                        "id": str(uuid4()),
                        "vector": vector,
                        "payload": {
                            **base_payload,
                            "section_number": str(index),
                            "section_title": "",
                            "text": chunk_text,
                        },
                    }
                )
        if points:
            self.retriever.upsert(points)
        return IngestStats(documents=len(documents), chunks=len(points))
