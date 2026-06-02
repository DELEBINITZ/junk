"""Admin endpoints: corpus ingestion + the action-approval inbox.

Two analyst-only capabilities live here:
  * INGESTION — load documents into a corpus so the RAG retrievers can later find
    them. Strictly org-scoped: the tenant comes from the token (SecurityContext),
    never the request body, so you can only ever ingest into your own org.
  * the APPROVAL INBOX — the human-in-the-loop half of the action gate. Recall
    (from contracts.py) that side-effecting tools don't run inline; they queue a
    pending action. These endpoints let an analyst LIST those and approve/reject
    them. v1 is a read-only product with no executable handlers yet, so approve
    just records intent; a later stage wires ``ActionHandler.execute``.

Every endpoint is guarded by ``require_role("analyst")`` — a FastAPI dependency
that 403s anyone below analyst, enforcing RBAC at the edge before the handler runs.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request

from app.core.api.deps import get_services, require_role
from app.core.api.schemas import IngestRequest, IngestResponse
from app.core.contracts import ToolContext
from app.core.errors import NotFound
from app.core.ingestion import IngestDocument
from app.core.security.context import SecurityContext

router = APIRouter(prefix="/v1", tags=["admin"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, request: Request,
                 sc: SecurityContext = Depends(require_role("analyst"))) -> IngestResponse:
    """Ingest a batch of documents into a corpus, scoped to the caller's org.

    We build a ToolContext from the VERIFIED identity (sc) — note org_id/roles
    come from the token, not the body — and pass it down so the pipeline stamps
    every resulting chunk with the right tenant. Fresh trace/request ids tie this
    operation together in observability. The audit record leaves a trail of who
    ingested what, into which collection."""
    services = get_services(request)
    ctx = ToolContext(org_id=sc.org_id, user_id=sc.user_id, roles=sc.roles,
                      trace_id=uuid.uuid4().hex, request_id=uuid.uuid4().hex, deps=services.deps)
    docs = [IngestDocument(**d.model_dump()) for d in body.documents]
    stats = await services.ingestion.ingest_documents(ctx, body.collection, docs, chunk=body.chunk)
    await services.audit.record(org_id=sc.org_id, user_id=sc.user_id, event="ingest",
                                collection=body.collection, documents=len(docs))
    return IngestResponse(documents=stats.documents, chunks=stats.chunks)


@router.get("/approvals")
async def list_approvals(request: Request,
                         sc: SecurityContext = Depends(require_role("analyst"))) -> list[dict]:
    """List this org's pending side-effecting actions awaiting approval. Scoped by
    ``sc.org_id`` so each tenant sees only its own inbox."""
    services = get_services(request)
    return services.action_gate.list_pending(sc.org_id)


@router.post("/approvals/{approval_id}/approve")
async def approve(approval_id: str, request: Request,
                  sc: SecurityContext = Depends(require_role("analyst"))) -> dict:
    """Approve one pending action. The gate call is org-scoped, so an analyst can
    only approve actions in their OWN org; an unknown/foreign id resolves to None
    and we 404 (never reveal another tenant's item). Approvals are audited, and
    ``.public()`` returns the client-safe view of the decision."""
    services = get_services(request)
    r = await services.action_gate.approve(sc.org_id, approval_id, sc.user_id)
    if not r:
        raise NotFound("approval not found")
    await services.audit.record(org_id=sc.org_id, user_id=sc.user_id, event="approval_approved",
                                approval_id=approval_id)
    return r.public()


@router.post("/approvals/{approval_id}/reject")
async def reject(approval_id: str, request: Request,
                 sc: SecurityContext = Depends(require_role("analyst"))) -> dict:
    """Reject one pending action — the mirror of approve, with the same org
    scoping, 404-on-miss, and audit trail."""
    services = get_services(request)
    r = await services.action_gate.reject(sc.org_id, approval_id, sc.user_id)
    if not r:
        raise NotFound("approval not found")
    await services.audit.record(org_id=sc.org_id, user_id=sc.user_id, event="approval_rejected",
                                approval_id=approval_id)
    return r.public()


__all__ = ["router"]
