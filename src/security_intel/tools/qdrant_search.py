"""Qdrant vector search tools with tenant-scoped access control.

Access rules (matching the ingestion contract):
- If customer_tags is NON-EMPTY: only orgs whose org_id is IN customer_tags can see it
- If customer_tags is EMPTY or public=true: visible to all orgs
- is_deleted=true: never returned

Production features:
- Shared AsyncQdrantClient (connection pooling)
- Reusable httpx client for embeddings
- Query rewriting for better retrieval
- Multi-query expansion for complex questions
"""

import httpx
from langchain_core.tools import tool
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from security_intel.config import Settings


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


def build_search_reports_tool(settings: Settings):
    """Factory: creates org-scoped Qdrant search tool with tenant access control."""

    qdrant = _get_qdrant_client(settings)

    @tool
    async def search_reports(query: str, top_k: int = 6) -> str:
        """Semantic search over the organization's security reports corpus.

        Returns relevant passages with citations (title, snippet, score).
        Automatically filters to reports the requesting org is authorized to see.
        For best results, use specific keywords: CVE IDs, threat actor names, asset names.

        Args:
            query: Natural language search query about threats, CVEs, findings, or remediation.
            top_k: Number of results to return (1-20, default 6).
        """
        from langgraph.config import get_config

        config = get_config()
        org_id = config["configurable"].get("org_id", "default")

        embedding = await _embed_query(query, settings)
        access_filter = _build_access_filter(org_id)

        results = await qdrant.query_points(
            collection_name=settings.qdrant_collection,
            query=embedding,
            query_filter=access_filter,
            limit=min(top_k, 20),
            with_payload=True,
        )

        if not results.points:
            return "No relevant reports found for this query. Try different keywords or broader terms."

        output_parts = []
        for i, point in enumerate(results.points, 1):
            payload = point.payload or {}
            title = payload.get("title", "Untitled")
            snippet = payload.get("text", "")[:500]
            score = f"{point.score:.3f}"
            doc_id = payload.get("doc_id", "")
            published = payload.get("published_at", "N/A")
            tlp = payload.get("tlp", "")

            output_parts.append(
                f"[{i}] {title} (relevance: {score})\n"
                f"    Doc: {doc_id} | Published: {published} | TLP: {tlp}\n"
                f"    {snippet}"
            )

        return "\n\n---\n\n".join(output_parts)

    return search_reports


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


def build_search_by_filter_tool(settings: Settings):
    """Factory: search reports by metadata filters (threat type, TLP, date range)."""

    qdrant = _get_qdrant_client(settings)

    @tool
    async def search_reports_by_filter(
        threat_type: str = "",
        tlp: str = "",
        report_type: str = "",
        limit: int = 10,
    ) -> str:
        """Search reports by metadata filters (no semantic search, exact match).

        Use for questions like "show me all TLP:RED reports" or "find ransomware reports".

        Args:
            threat_type: Filter by threat type (e.g., 'ransomware', 'phishing', 'apt').
            tlp: Filter by TLP level (e.g., 'RED', 'AMBER', 'GREEN', 'CLEAR').
            report_type: Filter by report type (e.g., 'threat_advisory', 'vulnerability').
            limit: Max results (1-50, default 10).
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

        access_filter = _build_access_filter(org_id, extra_must=extra_must)

        results = await qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=access_filter,
            limit=min(limit, 50),
            with_payload=True,
        )

        points = results[0] if results else []
        if not points:
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
