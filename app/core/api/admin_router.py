"""Admin endpoints: corpus ingestion + the action-approval inbox.

Ingestion is org-scoped (org from the token, never the body). The approval inbox
realizes the human-in-the-loop gate: side-effecting tools land here, an analyst
approves/rejects. v1 ships no executable handlers (read-only product), so approve
marks intent; Stage-D wires ``ActionHandler.execute``.
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
    services = get_services(request)
    return services.action_gate.list_pending(sc.org_id)


@router.post("/approvals/{approval_id}/approve")
async def approve(approval_id: str, request: Request,
                  sc: SecurityContext = Depends(require_role("analyst"))) -> dict:
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
    services = get_services(request)
    r = await services.action_gate.reject(sc.org_id, approval_id, sc.user_id)
    if not r:
        raise NotFound("approval not found")
    await services.audit.record(org_id=sc.org_id, user_id=sc.user_id, event="approval_rejected",
                                approval_id=approval_id)
    return r.public()


__all__ = ["router"]
