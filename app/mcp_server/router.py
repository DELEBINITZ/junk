"""MCP HTTP endpoint protected by the same JWT dependency as the API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app.auth.dependencies import require_user
from app.db.repository import DataStore, get_store
from app.domain import User
from app.mcp_server.protocol import handle_mcp_request


router = APIRouter(tags=["mcp"])


@router.post("/mcp")
def mcp_endpoint(
    payload: dict,
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Route JSON-RPC MCP requests through the authenticated protocol handler."""

    response = handle_mcp_request(payload, user, store)
    if response is None:
        return Response(status_code=status.HTTP_202_ACCEPTED)
    return response
