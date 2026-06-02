"""Authorized chunk ranking primitives for RAG search."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re

from app.domain import Chunk, Document
from app.rag.embeddings import EmbeddingProvider, cosine_similarity

logger = logging.getLogger(__name__)
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "of",
    "or",
    "search",
    "show",
    "the",
    "to",
    "what",
    "with",
}


@dataclass(slots=True)
class SearchHit:
    document: Document
    chunk: Chunk
    score: float


def search_chunks(
    query: str,
    chunks: list[Chunk],
    documents: dict[str, Document],
    embedder: EmbeddingProvider,
    top_k: int = 5,
) -> list[SearchHit]:
    """Rank the supplied chunks with a small hybrid retrieval score.

    The caller is responsible for passing only authorized chunks. Keeping that
    contract explicit makes cross-tenant filtering happen before vector ranking.

    The default test/demo embedder is a deterministic hashing embedder, so this
    function adds lexical boosts for exact contract IDs and clause-title terms.
    That keeps the PoC faithful to a production RAG design without requiring a
    model download for every test run.
    """

    query_embedding = embedder.embed(query)
    query_terms = _terms(query)
    query_lower = query.lower()
    hits = []
    for chunk in chunks:
        if chunk.document_id not in documents:
            continue
        document = documents[chunk.document_id]
        semantic_score = cosine_similarity(query_embedding, chunk.embedding)
        lexical_score = _lexical_overlap(query_terms, _terms(chunk.text))
        title_score = _lexical_overlap(query_terms, _terms(str(chunk.metadata.get("section_title", ""))))
        contract_boost = 0.08 if document.contract_id.lower() in query_lower else 0.0
        score = semantic_score + (0.12 * lexical_score) + (0.45 * title_score) + contract_boost
        hits.append(SearchHit(document=document, chunk=chunk, score=score))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    logger.debug(
        "rag.search.rank_complete",
        extra={
            "candidate_chunks": len(hits),
            "top_k": top_k,
            "top_hits": [
                {
                    "contract_id": hit.document.contract_id,
                    "section_number": hit.chunk.metadata.get("section_number"),
                    "section_title": hit.chunk.metadata.get("section_title"),
                    "score": round(hit.score, 4),
                }
                for hit in hits[:top_k]
            ],
        },
    )
    return hits[:top_k]


def _terms(text: str) -> set[str]:
    return {token for token in TOKEN.findall(text.lower()) if token not in STOP_WORDS and len(token) > 1}


def _lexical_overlap(query_terms: set[str], candidate_terms: set[str]) -> float:
    if not query_terms or not candidate_terms:
        return 0.0
    return len(query_terms.intersection(candidate_terms)) / math.sqrt(len(query_terms) * len(candidate_terms))
