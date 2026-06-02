"""HTTP-facing wrappers around RBAC decisions.

`permissions.py` answers yes/no questions. This module translates those answers
into consistent API errors for routers.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from app.db.repository import DataStore
from app.domain import Document, User
from app.rbac.permissions import can_delete_document, can_query_document, can_read_document


def require_document(store: DataStore, document_id: str) -> Document:
    """Load a non-deleted document or return a not-found error."""

    document = store.documents.get(document_id)
    if document is None or document.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


def require_read_access(user: User, document: Document, store: DataStore) -> None:
    """Raise 403 unless the user can read the document."""

    if not can_read_document(user, document, store):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Document access denied")


def require_query_access(user: User, document: Document, store: DataStore) -> None:
    """Raise 403 unless the document can be used in AI/RAG flows."""

    if not can_query_document(user, document, store):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Document query denied")


def require_delete_access(user: User, document: Document, store: DataStore) -> None:
    """Raise 403 unless the user can soft-delete the document."""

    if not can_delete_document(user, document, store):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Document delete denied")
