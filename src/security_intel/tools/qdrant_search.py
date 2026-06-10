"""Qdrant vector search tools with tenant-scoped access control.

Access rules (matching the ingestion contract):
- If customer_tags is NON-EMPTY: only orgs whose org_id is IN customer_tags can see it
- If customer_tags is EMPTY or public=true: visible to all orgs
- is_deleted=true: never returned

Production features:
- Shared AsyncQdrantClient (connection pooling)
- Reusable httpx client for embeddings
- Query enrichment: multi-query fan-out, HyDE, step-back
- Reciprocal Rank Fusion for merging multi-query results
- Reranking via cross-encoder
"""

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
from langchain_core.tools import tool
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny, DatetimeRange, OrderBy,
)

from security_intel.config import Settings
from security_intel.observability.logging import get_logger

logger = get_logger("qdrant_search")


def _date_range_condition(days: int = 0, start_date: str = "", end_date: str = "") -> FieldCondition | None:
    """Build a published_at datetime-range filter (indexed as DATETIME in Qdrant).

    days>0 → "last N days" relative to the server's current UTC time (no LLM date
    math). start_date/end_date → absolute RFC3339/ISO bounds. Returns None if no
    temporal constraint was requested.
    """
    gte = lte = None
    if days and days > 0:
        gte = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    if start_date:
        gte = start_date
    if end_date:
        lte = end_date
    if gte is None and lte is None:
        return None
    return FieldCondition(key="published_at", range=DatetimeRange(gte=gte, lte=lte))


async def _latest_report_date(qdrant: AsyncQdrantClient, settings: Settings, access_filter: Filter) -> str | None:
    """Most recent published_at available to this org (for honest 'no recent reports' messages)."""
    try:
        points, _ = await qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=access_filter,
            limit=1,
            with_payload=True,
            order_by=OrderBy(key="published_at", direction="desc"),
        )
        if points:
            return (points[0].payload or {}).get("published_at")
    except Exception as e:
        logger.warning(f"latest-report-date lookup failed: {e}")
    return None


_qdrant_clients: dict[str, AsyncQdrantClient] = {}
_http_client: httpx.AsyncClient | None = None


def _get_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """Singleton Qdrant client per URL — avoids recreating connections."""
    key = settings.qdrant_url
    if key not in _qdrant_clients:
        _qdrant_clients[key] = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30,
        )
    return _qdrant_clients[key]


def _get_http_client() -> httpx.AsyncClient:
    """Reusable httpx client for embedding requests — avoids connection churn."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=10))
    return _http_client


def _build_access_filter(org_id: str, extra_must: list | None = None) -> Filter:
    """Standard tenant access filter used across all tools."""
    must = [FieldCondition(key="is_deleted", match=MatchValue(value=False))]
    if extra_must:
        must.extend(extra_must)

    return Filter(
        must=must,
        should=[
            FieldCondition(key="customer_tags", match=MatchAny(any=[org_id])),
            FieldCondition(key="public", match=MatchValue(value=True)),
        ],
    )


def build_search_reports_tool(settings: Settings, enricher=None):
    """Factory: creates org-scoped Qdrant search tool with query enrichment.

    When enricher is provided, queries are expanded via multi-query/HyDE/step-back
    strategies and results are merged using Reciprocal Rank Fusion before reranking.
    """

    qdrant = _get_qdrant_client(settings)

    @tool
    async def search_reports(query: str, top_k: int = 6, days: int = 0) -> str:
        """Semantic search over the organization's security reports corpus.

        Uses adaptive query enrichment: automatically expands broad queries into
        multiple search variants for better recall, while keeping precise queries fast.

        Args:
            query: Natural language search query about threats, CVEs, findings, or remediation.
            top_k: Number of results to return (1-20, default 6).
            days: If >0, only return reports published in the last N days (e.g. 30 for
                "last 30 days", 7 for "this week"). The cutoff is computed from the
                current date — use this for any time-bound request.
        """
        from langgraph.config import get_config

        config = get_config()
        org_id = config["configurable"].get("org_id", "default")
        date_cond = _date_range_condition(days=days)
        access_filter = _build_access_filter(org_id, extra_must=[date_cond] if date_cond else None)

        search_queries = [query]
        if enricher:
            try:
                enriched = await enricher.enrich(query)
                search_queries = enriched.search_queries
                logger.info(
                    f"Enriched query: strategy={enriched.strategy.value}, "
                    f"variants={len(search_queries)}"
                )
            except Exception as e:
                logger.warning(f"Query enrichment failed ({e}), using original query")

        per_query_limit = min(top_k * 2, 30) if len(search_queries) > 1 else min(top_k, 20)
        if settings.reranker_enabled:
            per_query_limit = min(top_k * settings.reranker_overfetch_multiplier, 60)

        all_passages = await _fan_out_search(
            qdrant, search_queries, access_filter, per_query_limit, settings
        )

        if not all_passages:
            if date_cond:
                # Time-bound search came back empty — be explicit about the window
                # and the most recent report we DO have, instead of silently
                # returning nothing or (worse) stale reports as if they were recent.
                latest = await _latest_report_date(qdrant, settings, _build_access_filter(org_id))
                cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
                msg = f"No reports published in the last {days} days (since {cutoff})."
                if latest:
                    msg += f" The most recent report available is dated {latest}."
                return msg
            return "No relevant reports found for this query. Try different keywords or broader terms."

        if len(search_queries) > 1 and len(all_passages) > 1:
            merged = _reciprocal_rank_fusion(all_passages, k=60)
        else:
            merged = all_passages[0] if all_passages else []

        if not merged:
            return "No relevant reports found for this query. Try different keywords or broader terms."

        if settings.reranker_enabled:
            reranked = await _rerank(query, merged, settings)

            # Drop clearly-irrelevant passages by rerank relevance. Only when the
            # reranker actually scored them (skip when it degraded to vector order),
            # and always keep at least the top hit so we never return empty.
            threshold = settings.reranker_score_threshold
            if threshold > 0 and any("rerank_score" in p for p in reranked):
                kept = [p for p in reranked if p.get("rerank_score", 0) >= threshold]
                reranked = kept or reranked[:1]

            final_k = settings.reranker_top_n if settings.reranker_top_n > 0 else top_k
            reranked = reranked[:min(final_k, 20)]

            return _format_passages(reranked)

        final = merged[:min(top_k, 20)]
        return _format_passages(final)

    return search_reports


def _format_passages(passages: list[dict]) -> str:
    """Render passages for the LLM using report content only.

    Deliberately excludes internal fields (doc_id, point_id, TLP, and
    vector/rerank/rrf scores) so they cannot leak into user-facing answers.
    Title and publish date are legitimate report metadata, kept for grounding.
    """
    parts = []
    for i, p in enumerate(passages, 1):
        snippet = p.get("text", "")[:500]
        published = p.get("published_at", "")
        header = f"[{i}] {p.get('title', 'Untitled')}"
        if published and published != "N/A":
            header += f" (published {published})"
        parts.append(f"{header}\n    {snippet}")
    return "\n\n---\n\n".join(parts)


async def _fan_out_search(
    qdrant: AsyncQdrantClient,
    queries: list[str],
    access_filter: Filter,
    per_query_limit: int,
    settings: Settings,
) -> list[list[dict]]:
    """Execute parallel vector searches for all query variants.

    Returns list of passage lists (one per query variant).
    """

    async def _search_single(q: str) -> list[dict]:
        embedding = await _embed_query(q, settings)
        results = await qdrant.query_points(
            collection_name=settings.qdrant_collection,
            query=embedding,
            query_filter=access_filter,
            limit=per_query_limit,
            with_payload=True,
        )
        passages = []
        for point in results.points:
            payload = point.payload or {}
            passages.append({
                "text": payload.get("text", ""),
                "title": payload.get("title", "Untitled"),
                "doc_id": payload.get("doc_id", ""),
                "published_at": payload.get("published_at", "N/A"),
                "tlp": payload.get("tlp", ""),
                "vector_score": point.score,
                "point_id": str(point.id),
            })
        return passages

    results = await asyncio.gather(*[_search_single(q) for q in queries], return_exceptions=True)
    valid = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning(f"Fan-out search variant {i} failed: {r}")
        elif r:
            valid.append(r)
    return valid


def _reciprocal_rank_fusion(
    result_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) across all lists where document appears.
    Higher k = more emphasis on lower-ranked documents (smoother distribution).
    """
    scores: dict[str, float] = {}
    passages: dict[str, dict] = {}

    for ranked_list in result_lists:
        for rank, passage in enumerate(ranked_list):
            doc_key = f"{passage['doc_id']}:{passage['text'][:100]}"
            scores[doc_key] = scores.get(doc_key, 0) + 1.0 / (k + rank + 1)
            if doc_key not in passages:
                passages[doc_key] = passage

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    merged = []
    for key in sorted_keys:
        passage = passages[key]
        passage["rrf_score"] = scores[key]
        merged.append(passage)

    return merged


def build_get_report_metadata_tool(settings: Settings):
    """Factory: creates tool to get report metadata by doc_id (org-scoped)."""

    qdrant = _get_qdrant_client(settings)

    @tool
    async def get_report_metadata(doc_id: str) -> str:
        """Get metadata for a specific security report by its document ID.

        Args:
            doc_id: The document/report identifier.
        """
        from langgraph.config import get_config

        config = get_config()
        org_id = config["configurable"].get("org_id", "default")

        access_filter = _build_access_filter(
            org_id,
            extra_must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))],
        )

        results = await qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=access_filter,
            limit=1,
            with_payload=True,
        )

        points = results[0] if results else []
        if not points:
            return f"No report found with ID '{doc_id}' (or access denied)."

        payload = points[0].payload or {}
        return (
            f"Title: {payload.get('title', 'N/A')}\n"
            f"Doc ID: {doc_id}\n"
            f"Published: {payload.get('published_at', 'N/A')}\n"
            f"TLP: {payload.get('tlp', 'N/A')}\n"
            f"Report Type: {payload.get('report_type', 'N/A')}\n"
            f"Reliability: {payload.get('reliability_rating', 'N/A')}\n"
            f"IOCs Count: {payload.get('iocs_count', 'N/A')}\n"
            f"Chunk: {payload.get('section', '')} of {payload.get('total_chunks', '?')}\n"
            f"Created: {payload.get('report_created_ts', 'N/A')}"
        )

    return get_report_metadata


def build_get_report_content_tool(settings: Settings):
    """Factory: fetch the full text of one report by doc_id (org-scoped).

    Use for "summarize report <ID>" — pulls all chunks for the document and
    returns them in section order so the agent summarizes from real content
    instead of semantic-searching a numeric ID (which retrieves poorly).
    """

    qdrant = _get_qdrant_client(settings)

    @tool
    async def get_report_content(doc_id: str, max_chars: int = 6000) -> str:
        """Get the full text content of a specific security report by its document ID.

        Use this to summarize or answer questions about ONE specific report when you
        already know its ID (e.g., "summarize report 2024020924468").

        Args:
            doc_id: The document/report identifier.
            max_chars: Max characters of combined content to return (default 6000).
        """
        from langgraph.config import get_config

        config = get_config()
        org_id = config["configurable"].get("org_id", "default")
        access_filter = _build_access_filter(
            org_id,
            extra_must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))],
        )

        points, _ = await qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=access_filter,
            limit=100,
            with_payload=True,
        )
        if not points:
            return f"No report found with ID '{doc_id}' (or access denied)."

        def _section_key(p):
            payload = p.payload or {}
            sec = payload.get("section", payload.get("chunk_index", 0))
            try:
                return int(sec)
            except (TypeError, ValueError):
                return 0

        ordered = sorted(points, key=_section_key)
        title = (ordered[0].payload or {}).get("title", "Untitled")

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
        return f"Report: {title}\n\n{body}"

    return get_report_content


def build_search_by_filter_tool(settings: Settings):
    """Factory: search reports by metadata filters (threat type, TLP, date range)."""

    qdrant = _get_qdrant_client(settings)

    @tool
    async def search_reports_by_filter(
        threat_type: str = "",
        tlp: str = "",
        report_type: str = "",
        limit: int = 10,
        days: int = 0,
    ) -> str:
        """Search reports by metadata filters (no semantic search, exact match).

        Use for questions like "show me all TLP:RED reports" or "find ransomware reports".

        Args:
            threat_type: Filter by threat type (e.g., 'ransomware', 'phishing', 'apt').
            tlp: Filter by TLP level (e.g., 'RED', 'AMBER', 'GREEN', 'CLEAR').
            report_type: Filter by report type (e.g., 'threat_advisory', 'vulnerability').
            limit: Max results (1-50, default 10).
            days: If >0, only reports published in the last N days (cutoff from current date).
        """
        from langgraph.config import get_config

        config = get_config()
        org_id = config["configurable"].get("org_id", "default")

        extra_must = []
        if threat_type:
            extra_must.append(
                FieldCondition(key="threat_types", match=MatchValue(value=threat_type))
            )
        if tlp:
            extra_must.append(
                FieldCondition(key="tlp", match=MatchValue(value=tlp.upper()))
            )
        if report_type:
            extra_must.append(
                FieldCondition(key="report_type", match=MatchValue(value=report_type))
            )
        date_cond = _date_range_condition(days=days)
        if date_cond:
            extra_must.append(date_cond)

        access_filter = _build_access_filter(org_id, extra_must=extra_must)

        results = await qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=access_filter,
            limit=min(limit, 50),
            with_payload=True,
        )

        points = results[0] if results else []
        if not points:
            if days > 0:
                latest = await _latest_report_date(qdrant, settings, _build_access_filter(org_id))
                cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
                msg = f"No reports published in the last {days} days (since {cutoff})."
                if latest:
                    msg += f" The most recent report available is dated {latest}."
                return msg
            return "No reports found matching the given filters."

        seen_docs = set()
        output_parts = []
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id", "")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            output_parts.append(
                f"- {payload.get('title', 'Untitled')} "
                f"(TLP: {payload.get('tlp', '?')}, Type: {payload.get('report_type', '?')}, "
                f"Published: {payload.get('published_at', '?')})"
            )

        return f"Found {len(output_parts)} reports:\n" + "\n".join(output_parts)

    return search_reports_by_filter


async def _rerank(query: str, passages: list[dict], settings: Settings) -> list[dict]:
    """Rerank passages using TEI /rerank endpoint (Qwen3-Reranker).

    Args:
        query: Original search query.
        passages: List of dicts with at least 'text' key.
        settings: App settings with reranker_base_url.

    Returns:
        Passages reordered by reranker score (descending), with 'rerank_score' added.
    """
    if not passages:
        return passages

    client = _get_http_client()
    texts = [p.get("text", "")[:2000] for p in passages]

    try:
        # Payload matches the reranker's accepted schema: {query, texts}. Extra
        # fields (e.g. "truncate") trigger 422 Unprocessable Entity on this server.
        resp = await client.post(
            f"{settings.reranker_base_url}/rerank",
            json={"query": query, "texts": texts},
            timeout=15,
        )
        resp.raise_for_status()
        rankings = resp.json()
        # Some servers wrap results: {"results": [...]} — accept either shape.
        if isinstance(rankings, dict):
            rankings = rankings.get("results") or rankings.get("data") or []

        for item in rankings:
            idx = item["index"]
            # Server indexes into the texts we sent; guard against out-of-range.
            if not isinstance(idx, int) or idx < 0 or idx >= len(passages):
                continue
            score = item.get("score", item.get("relevance_score", 0))
            passages[idx]["rerank_score"] = score
    except (httpx.HTTPError, httpx.ConnectError, KeyError, TypeError, ValueError) as e:
        # Reranker down/unreachable or unexpected response — degrade gracefully to
        # the existing order (vector score / RRF) instead of crashing the agent.
        logger.warning(f"Rerank failed ({e}), returning passages without reranking")
        return passages

    return sorted(passages, key=lambda p: p.get("rerank_score", 0), reverse=True)


async def _embed_query(query: str, settings: Settings) -> list[float]:
    """Embed a query using the TEI embedding endpoint.

    Uses shared httpx client for connection reuse.
    """
    client = _get_http_client()

    try:
        resp = await client.post(
            f"{settings.embedding_base_url}/embed",
            json={"inputs": [query]},
        )
        resp.raise_for_status()
        data = resp.json()
        return data[0]
    except (httpx.HTTPStatusError, KeyError):
        pass

    resp = await client.post(
        f"{settings.embedding_base_url}/v1/embeddings",
        json={"input": query, "model": settings.embedding_model},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]
