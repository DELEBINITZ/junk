"""AI query endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from app.agent.workflow import run_agent_query
from app.auth.dependencies import require_user
from app.db.repository import DataStore, get_store
from app.domain import User


router = APIRouter(prefix="/ai", tags=["ai"])


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    organization_id: str | None = None


@router.post("/query")
def query_ai(
    payload: QueryRequest,
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Run the agent workflow for authenticated non-viewer users."""

    if payload.organization_id is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="organization_id is derived from the JWT and cannot be supplied",
        )
    if user.role == "viewer":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Viewers cannot query AI")
    result = run_agent_query(payload.query, user, store)
    if "error" in result:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result)
    return result


@router.get("/query/{query_id}/events")
def query_events(
    query_id: str,
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Return the stored trace for a query in the caller's organization."""

    record = store.queries.get(query_id)
    if record is None or record.organization_id != user.organization_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Query not found")
    return {"query_id": query_id, "status": record.status, "plan": record.plan, "result": record.result}
