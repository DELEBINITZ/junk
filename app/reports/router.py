"""Report persistence endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.db.repository import DataStore, get_store
from app.domain import Report, User, new_id
from app.auth.dependencies import require_user
from app.rbac.permissions import can_create_report


router = APIRouter(prefix="/reports", tags=["reports"])


class CreateReportRequest(BaseModel):
    title: str
    query: str
    result: dict


@router.get("")
def list_reports(user: User = Depends(require_user), store: DataStore = Depends(get_store)):
    """List reports scoped to the authenticated user's organization."""

    return {
        "reports": [
            report
            for report in store.reports.values()
            if report.organization_id == user.organization_id
        ]
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_report(
    payload: CreateReportRequest,
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Create a report for admin/analyst users in their own organization."""

    if not can_create_report(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot create reports")
    report = Report(
        id=new_id(),
        organization_id=user.organization_id,
        created_by=user.id,
        title=payload.title,
        query=payload.query,
        result=payload.result,
    )
    store.add_report(report)
    return report
