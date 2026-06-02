"""Reports module tools.

Retrieval (RAG) is provided by the module's retriever; these are *targeted*
structured tools over the same org-scoped corpus. They pull the live pipeline
from ``ctx.deps.rag`` — no global state, fully tenant-isolated.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.contracts import ToolContext, ToolError, ToolResult, tool
from app.core.rag.vector_store import SearchFilters


class MetadataArgs(BaseModel):
    doc_id: str = Field(description="The report/document id, e.g. 'R-1042'.")


class ExpiringArgs(BaseModel):
    query: str = Field(default="expiring certificate renewal deadline", description="What to scan for.")
    horizon_days: int = Field(default=90, description="Look-ahead window in days.")


@tool(
    name="get_report_metadata",
    description="Fetch metadata (title, date, section count) for a specific report by its doc_id.",
    args_schema=MetadataArgs,
    rbac_role="viewer",
)
async def get_report_metadata(args: MetadataArgs, ctx: ToolContext):
    chunks = await ctx.deps.rag.retrieve(
        args.doc_id, collection="reports_kb", ctx=ctx, top_k=25,
        filters=SearchFilters(doc_ids=[args.doc_id]), apply_time_filters=False,
    )
    if not chunks:
        return ToolError(code="not_found", message=f"no report '{args.doc_id}' for this org")
    first = chunks[0]
    return ToolResult(
        data={"doc_id": args.doc_id, "title": first.title,
              "published_at": first.published_at, "sections": len(chunks),
              "source": first.source},
        citations=[first.to_citation()],
    )


@tool(
    name="find_expiring_items",
    description="Surface time-sensitive items (e.g. expiring certificates, deadlines) from the reports corpus.",
    args_schema=ExpiringArgs,
    rbac_role="viewer",
)
async def find_expiring_items(args: ExpiringArgs, ctx: ToolContext):
    chunks = await ctx.deps.rag.retrieve(
        args.query, collection="reports_kb", ctx=ctx, top_k=6, apply_time_filters=False,
    )
    items = [{"doc_id": c.doc_id, "title": c.title, "snippet": c.text[:200],
              "published_at": c.published_at} for c in chunks]
    return ToolResult(
        data={"items": items, "count": len(items), "horizon_days": args.horizon_days},
        citations=[c.to_citation() for c in chunks[:3]],
    )


TOOLS = (get_report_metadata, find_expiring_items)

__all__ = ["TOOLS", "get_report_metadata", "find_expiring_items"]
