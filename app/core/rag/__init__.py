"""Shared retrieval: embeddings, org-filtered vector store, reranker, pipeline."""

from app.core.rag.citations import GroundednessReport, verify_groundedness
from app.core.rag.embeddings import Embedder, build_embedder
from app.core.rag.pipeline import (
    CollectionRetriever,
    IndexItem,
    RetrievalPipeline,
    build_rag,
    build_vector_store,
)
from app.core.rag.reranker import Reranker, build_reranker
from app.core.rag.vector_store import SearchFilters, VectorRecord, VectorStore

__all__ = [
    "Embedder",
    "build_embedder",
    "VectorStore",
    "VectorRecord",
    "SearchFilters",
    "Reranker",
    "build_reranker",
    "RetrievalPipeline",
    "CollectionRetriever",
    "IndexItem",
    "build_rag",
    "build_vector_store",
    "GroundednessReport",
    "verify_groundedness",
]
