"""Centralized RBAC rules used by both API endpoints and MCP tools.

The most important convention in this module is that tenant isolation is checked
before role-specific permissions. A user must first be in the same organization
as the document; only then do admin/analyst/viewer rules apply.
"""

from __future__ import annotations

from app.db.repository import DataStore
from app.domain import AccessLevel, Document, User


ACCESS_ORDER: dict[AccessLevel, int] = {"read": 1, "query": 2, "edit": 3}


def can_manage_users(user: User, organization_id: str) -> bool:
    return user.organization_id == organization_id and user.role == "admin"


def can_upload_document(user: User) -> bool:
    return user.role in {"admin", "analyst"}


def can_configure_guardrails(user: User, organization_id: str) -> bool:
    return user.organization_id == organization_id and user.role == "admin"


def can_create_report(user: User) -> bool:
    return user.role in {"admin", "analyst"}


def can_read_document(user: User, document: Document, store: DataStore) -> bool:
    """Return whether a user can read document text and metadata."""

    if document.deleted_at is not None:
        return False
    if user.organization_id != document.organization_id:
        return False
    if user.role == "admin":
        return True
    if user.role == "analyst":
        return document.uploaded_by == user.id or _has_share(
            document, user, store, ["read", "query", "edit"]
        )
    if user.role == "viewer":
        return _has_share(document, user, store, ["read", "query", "edit"])
    return False


def can_query_document(user: User, document: Document, store: DataStore) -> bool:
    """Return whether a document may participate in AI/RAG/MCP query results.

    Query access is intentionally stricter than read access: viewers may read
    explicitly shared documents, but cannot invoke the AI workflow.
    """

    if document.deleted_at is not None:
        return False
    if user.organization_id != document.organization_id:
        return False
    if user.role == "admin":
        return True
    if user.role == "analyst":
        return document.uploaded_by == user.id or _has_share(
            document, user, store, ["query", "edit"]
        )
    return False


def can_delete_document(user: User, document: Document, store: DataStore) -> bool:
    """Return whether a user can soft-delete a document."""

    if document.deleted_at is not None:
        return False
    if user.organization_id != document.organization_id:
        return False
    if user.role == "admin":
        return True
    if user.role == "analyst":
        return document.uploaded_by == user.id or _has_share(document, user, store, ["edit"])
    return False


def can_share_document(user: User, document: Document) -> bool:
    if user.organization_id != document.organization_id:
        return False
    return user.role == "admin" or document.uploaded_by == user.id


def readable_documents(user: User, store: DataStore) -> list[Document]:
    """List documents visible to the user after tenant and role checks."""

    return [
        document
        for document in store.documents_for_org(user.organization_id)
        if can_read_document(user, document, store)
    ]


def queryable_documents(user: User, store: DataStore) -> list[Document]:
    """List documents that may be used by AI tools and vector retrieval."""

    return [
        document
        for document in store.documents_for_org(user.organization_id)
        if can_query_document(user, document, store)
    ]


def _has_share(
    document: Document,
    user: User,
    store: DataStore,
    accepted_access_levels: list[AccessLevel],
) -> bool:
    """Check document share level against the access levels accepted by a caller."""

    share = store.share_for(document.id, user.id)
    return bool(share and share.access_level in accepted_access_levels)
