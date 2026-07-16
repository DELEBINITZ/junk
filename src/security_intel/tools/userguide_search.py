"""Qdrant search tools for the FortiRecon product user guide.

Queries the dedicated documentation collection (``settings.user_guide_collection``,
default ``user_guide_kb``) — kept separate from the threat-report corpus so
how-to / navigation pages never surface in report search and vice-versa.

Reuses the shared embedding / client / rerank helpers from ``qdrant_search`` so
the docs corpus is embedded and queried with the SAME model (TEI Qwen3-Embedding,
dim 2560) and access-control contract as reports:
- ``is_deleted=true`` is never returned
- a point is visible when ``public=true`` OR its ``customer_tags`` includes org_id
  (product docs are ingested public, so they are visible to every org)
"""

from langchain_core.tools import tool
from qdrant_client.models import FieldCondition, MatchValue

from security_intel.config import Settings
from security_intel.observability.logging import get_logger
from security_intel.tools.qdrant_search import (
    _build_access_filter,
    _embed_query,
    _get_qdrant_client,
    _reciprocal_rank_fusion,
    _rerank,
)

logger = get_logger("userguide_search")


def _format_doc_passages(passages: list[dict]) -> str:
    """Render doc passages for the LLM.

    Unlike report passages, doc title / heading / URL ARE user-facing and useful
    for grounding a walkthrough ("see the Dashboard page"), so they are kept.
    Internal scores and point ids are deliberately excluded.
    """
    parts = []
    for i, p in enumerate(passages, 1):
        snippet = p.get("text", "")[:700]
        title = p.get("title", "Untitled")
        heading = p.get("heading", "")
        url = p.get("url", "")
        breadcrumb = p.get("breadcrumb", "")
        # Lead with the nav location so the agent can tell the user WHERE this lives
        # ("Attack Surface Management > EASM > EASM Dashboard") for walkthroughs.
        header = f"[{i}] {breadcrumb}" if breadcrumb else f"[{i}] {title}"
        if heading and heading != title and heading not in breadcrumb:
            header += f" — {heading}"
        parts.append(f"{header}\n    {snippet}" + (f"\n    (source: {url})" if url else ""))
    return "\n\n---\n\n".join(parts)


def _passage_from_point(point) -> dict:
    payload = point.payload or {}
    return {
        "text": payload.get("text", ""),
        "title": payload.get("title", "Untitled"),
        "heading": payload.get("heading", ""),
        "breadcrumb": payload.get("breadcrumb", ""),
        "url": payload.get("url", ""),
        "doc_id": payload.get("doc_id", ""),
        "vector_score": point.score,
    }


def build_search_user_guide_tool(settings: Settings, enricher=None):
    """Factory: semantic search over the FortiRecon user-guide corpus.

    When ``enricher`` is provided, broad queries are expanded into multiple
    variants and merged with Reciprocal Rank Fusion before optional reranking —
    same recall strategy as report search.
    """

    qdrant = _get_qdrant_client(settings)
    collection = settings.user_guide_collection

    @tool
    async def search_user_guide(query: str, top_k: int = 6) -> str:
        """Search the FortiRecon product user guide (documentation).

        Use for "how do I…", "where do I find…", "walk me through…", dashboard and
        feature explanations, configuration steps, and navigation questions about
        the FortiRecon platform itself.

        Args:
            query: Natural-language question about using the FortiRecon product.
            top_k: Number of passages to return (1-20, default 6).
        """
        from langgraph.config import get_config

        config = get_config()
        org_id = config["configurable"].get("org_id", "default")
        access_filter = _build_access_filter(org_id)

        search_queries = [query]
        if enricher:
            try:
                enriched = await enricher.enrich(query)
                search_queries = enriched.search_queries
            except Exception as e:
                logger.warning(f"Query enrichment failed ({e}), using original query")

        if settings.reranker_enabled:
            per_query_limit = min(top_k * settings.reranker_overfetch_multiplier, 60)
        else:
            per_query_limit = min(top_k * 2, 30) if len(search_queries) > 1 else min(top_k, 20)

        result_lists: list[list[dict]] = []
        for q in search_queries:
            try:
                embedding = await _embed_query(q, settings)
                results = await qdrant.query_points(
                    collection_name=collection,
                    query=embedding,
                    query_filter=access_filter,
                    limit=per_query_limit,
                    with_payload=True,
                )
                result_lists.append([_passage_from_point(p) for p in results.points])
            except Exception as e:
                logger.warning(f"User-guide search variant failed: {e}")

        result_lists = [r for r in result_lists if r]
        if not result_lists:
            return (
                "I couldn't find anything about that in the FortiRecon user guide. "
                "Try rephrasing, or name the specific feature/page you're asking about."
            )

        if len(result_lists) > 1:
            merged = _reciprocal_rank_fusion(result_lists, k=60)
        else:
            merged = result_lists[0]

        if settings.reranker_enabled:
            merged = await _rerank(query, merged, settings)
            final_k = settings.reranker_top_n if settings.reranker_top_n > 0 else top_k
            merged = merged[: min(final_k, 20)]
        else:
            merged = merged[: min(top_k, 20)]

        return _format_doc_passages(merged)

    return search_user_guide


def build_get_user_guide_page_tool(settings: Settings):
    """Factory: fetch the full text of one user-guide page by its doc_id.

    Use after a search identifies the right page (e.g. the Dashboard page) to pull
    the complete walkthrough in section order instead of isolated snippets.
    """

    qdrant = _get_qdrant_client(settings)
    collection = settings.user_guide_collection

    @tool
    async def get_user_guide_page(doc_id: str, max_chars: int = 8000) -> str:
        """Get the full content of one FortiRecon user-guide page by its page id.

        Args:
            doc_id: The page identifier returned/implied by search results.
            max_chars: Max characters of combined content to return (default 8000).
        """
        from langgraph.config import get_config

        config = get_config()
        org_id = config["configurable"].get("org_id", "default")
        access_filter = _build_access_filter(
            org_id,
            extra_must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))],
        )

        points, _ = await qdrant.scroll(
            collection_name=collection,
            scroll_filter=access_filter,
            limit=200,
            with_payload=True,
        )
        if not points:
            return f"No user-guide page found with id '{doc_id}'."

        def _chunk_key(p):
            payload = p.payload or {}
            idx = (payload.get("metadata") or {}).get("chunk_index", 0)
            try:
                return int(idx)
            except (TypeError, ValueError):
                return 0

        ordered = sorted(points, key=_chunk_key)
        first = ordered[0].payload or {}
        title = first.get("title", "Untitled")
        url = first.get("url", "")
        breadcrumb = first.get("breadcrumb", "")

        parts, total = [], 0
        for p in ordered:
            text = (p.payload or {}).get("text", "")
            if not text:
                continue
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break

        body = "\n\n".join(parts)[:max_chars]
        header = f"Page: {breadcrumb or title}"
        if url:
            header += f"\nSource: {url}"
        return f"{header}\n\n{body}"

    return get_user_guide_page
