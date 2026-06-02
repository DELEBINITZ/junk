"""Reports module tools — the typed functions the agent can call for this module.

A "tool" here is a :class:`Tool` from contracts.py: a typed, MCP-exposable
function the agent (or a specialist) may invoke. The general way the reports
module answers a question is its RETRIEVER (broad RAG over the corpus); the two
tools below are *targeted* structured shortcuts over that SAME org-scoped corpus,
for jobs retrieval alone does poorly (look up one doc by id; surface expiring
items). Both are declared with the ``@tool`` decorator, which wraps the async
handler in a Tool the manifest then advertises.

Two contracts to notice as you read:
  * ERRORS-AS-DATA: a handler returns a :class:`ToolResult` (success) or a
    :class:`ToolError` (failure) — it must never raise into the agent. The Tool
    wrapper additionally catches any stray exception and converts it to a
    ToolError, so one bad tool can't crash a turn.
  * TENANT ISOLATION: tools take no org argument. They reach the live retrieval
    pipeline via ``ctx.deps.rag`` and the tenant via ``ctx`` (the trusted,
    token-derived ToolContext). There is NO global state, so a tool cannot be
    pointed at another org's data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.contracts import ToolContext, ToolError, ToolResult, tool
from app.core.rag.vector_store import SearchFilters


# Each tool declares its arguments as a pydantic model. The platform gets free
# validation AND a JSON schema from it: the schema is what is advertised to the
# LLM for function-calling, and the validation is what guarantees the handler only
# ever sees well-formed args (bad input becomes a ToolError("invalid_args")).
class MetadataArgs(BaseModel):
    doc_id: str = Field(description="The report/document id, e.g. 'R-1042'.")


class ExpiringArgs(BaseModel):
    # Both fields have defaults, so NEITHER is required. That matters for the
    # heuristic specialist planner (autocall_args in specialist.py): a tool whose
    # required args can't be inferred from the question is skipped, but this tool —
    # being fully defaultable, with a free-text ``query`` field — can be auto-invoked.
    query: str = Field(default="expiring certificate renewal deadline", description="What to scan for.")
    horizon_days: int = Field(default=90, description="Look-ahead window in days.")


# READ tool (rbac_role="viewer", not side-effecting). The decorator turns this
# async handler into a Tool named "get_report_metadata"; its ``description`` is
# what the LLM reads to decide when to call it, and ``args_schema`` is MetadataArgs.
@tool(
    name="get_report_metadata",
    description="Fetch metadata (title, date, section count) for a specific report by its doc_id.",
    args_schema=MetadataArgs,
    rbac_role="viewer",
)
async def get_report_metadata(args: MetadataArgs, ctx: ToolContext):
    # Retrieve via the shared pipeline on ``ctx.deps`` — note there is no org_id in
    # this call: the pipeline reads the tenant from ``ctx`` (token-derived), so the
    # ``doc_ids`` filter can only ever match documents the caller's org owns. We pull
    # all chunks of the one doc (top_k=25) just to count its sections.
    chunks = await ctx.deps.rag.retrieve(
        args.doc_id, collection="reports_kb", ctx=ctx, top_k=25,
        filters=SearchFilters(doc_ids=[args.doc_id]), apply_time_filters=False,
    )
    # FAILURE as a value, not an exception: unknown/forbidden doc -> a ToolError the
    # agent receives like any other result and reasons about, instead of crashing.
    if not chunks:
        return ToolError(code="not_found", message=f"no report '{args.doc_id}' for this org")
    first = chunks[0]
    # SUCCESS shape: structured ``data`` plus a Citation so the answer can ground
    # the claim in this exact source ([n] markers downstream map back to it).
    return ToolResult(
        data={"doc_id": args.doc_id, "title": first.title,
              "published_at": first.published_at, "sections": len(chunks),
              "source": first.source},
        citations=[first.to_citation()],
    )


# READ tool (viewer, not side-effecting). Because its args are fully defaultable
# AND it has a free-text ``query``, the heuristic planner can auto-invoke it on a
# broad question — making it the module's "what's coming up?" gatherer.
@tool(
    name="find_expiring_items",
    description="Surface time-sensitive items (e.g. expiring certificates, deadlines) from the reports corpus.",
    args_schema=ExpiringArgs,
    rbac_role="viewer",
)
async def find_expiring_items(args: ExpiringArgs, ctx: ToolContext):
    # Same tenant-safe retrieval (org comes from ``ctx``, not args). A semantic search
    # over the corpus for time-sensitive language; we keep the top few hits.
    chunks = await ctx.deps.rag.retrieve(
        args.query, collection="reports_kb", ctx=ctx, top_k=6, apply_time_filters=False,
    )
    items = [{"doc_id": c.doc_id, "title": c.title, "snippet": c.text[:200],
              "published_at": c.published_at} for c in chunks]
    # Return the structured list as ``data`` and the first few chunks as citations
    # (evidence). The agent never has to try/except — it just checks ok/data.
    return ToolResult(
        data={"items": items, "count": len(items), "horizon_days": args.horizon_days},
        citations=[c.to_citation() for c in chunks[:3]],
    )


# The tuple the manifest imports as ``tools=TOOLS``. This is the module's entire
# MCP/agent surface; ordering is irrelevant (the supervisor/specialist pick by need).
TOOLS = (get_report_metadata, find_expiring_items)

__all__ = ["TOOLS", "get_report_metadata", "find_expiring_items"]
