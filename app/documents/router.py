"""Document management endpoints with tenant-aware RBAC checks."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from app.audit.logger import log_event
from app.auth.dependencies import require_user
from app.db.repository import DataStore, get_store
from app.documents.ingestion import ingest_contract_text
from app.domain import AccessLevel, DocumentShare, User
from app.guardrails.pii import redact_pii
from app.rbac.permissions import (
    can_share_document,
    can_upload_document,
    readable_documents,
)
from app.rbac.policies import require_delete_access, require_document, require_read_access


router = APIRouter(prefix="/documents", tags=["documents"])


class UploadDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = "uploaded.txt"
    raw_text: str
    tags: list[str] = []
    organization_id: str | None = None


class ShareDocumentRequest(BaseModel):
    user_id: str
    access_level: AccessLevel


@router.get("")
def list_documents(user: User = Depends(require_user), store: DataStore = Depends(get_store)):
    """Return only documents visible to the authenticated user."""

    return {"documents": [_document_summary(document) for document in readable_documents(user, store)]}


@router.post("", status_code=status.HTTP_201_CREATED)
def upload_document(
    payload: UploadDocumentRequest,
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Upload a contract into the caller's organization after parsing metadata."""

    if payload.organization_id is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="organization_id is derived from the JWT and cannot be supplied",
        )
    if not can_upload_document(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Viewers cannot upload")
    document = ingest_contract_text(
        store,
        raw_text=payload.raw_text,
        filename=payload.filename,
        uploaded_by=user.id,
        tags=payload.tags,
    )
    if document.organization_id != user.organization_id:
        # Uploaded documents inherit tenant from extracted metadata. Rejecting a mismatch keeps
        # the request from smuggling a contract into another tenant.
        del store.documents[document.id]
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Contract tenant mismatch")
    log_event(store, user, "document.upload", "document", document.id, "success")
    return _document_summary(document)


@router.get("/{document_id}")
def get_document(
    document_id: str,
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Return redacted document text and sections after read authorization."""

    document = require_document(store, document_id)
    require_read_access(user, document, store)
    return {
        **_document_summary(document),
        "text": redact_pii(document.raw_text),
        "sections": [
            {
                "section_number": section.section_number,
                "section_title": section.section_title,
                "line_start": section.line_start,
                "line_end": section.line_end,
                "text": redact_pii(section.text),
            }
            for section in store.sections_for_document(document.id)
        ],
    }


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Soft-delete a document after delete authorization."""

    document = require_document(store, document_id)
    require_delete_access(user, document, store)
    document.deleted_at = datetime.now(UTC)
    log_event(store, user, "document.delete", "document", document.id, "success")
    return None


@router.post("/{document_id}/share")
def share_document(
    document_id: str,
    payload: ShareDocumentRequest,
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Share a document with another user in the same organization."""

    document = require_document(store, document_id)
    if not can_share_document(user, document):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot share document")
    target = store.users.get(payload.user_id)
    if target is None or target.organization_id != user.organization_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Share target must be in org")
    share = DocumentShare(
        document_id=document.id,
        user_id=target.id,
        access_level=payload.access_level,
    )
    store.add_share(share)
    log_event(store, user, "document.share", "document", document.id, "success")
    return {"document_id": document.id, "user_id": target.id, "access_level": share.access_level}


def _document_summary(document):
    return {
        "id": document.id,
        "organization_id": document.organization_id,
        "contract_id": document.contract_id,
        "title": document.title,
        "filename": document.filename,
        "uploaded_by": document.uploaded_by,
        "tags": document.tags,
        "metadata": document.metadata,
        "created_at": document.created_at.isoformat(),
    }
