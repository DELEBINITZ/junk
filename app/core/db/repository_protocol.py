"""The repository surface the application depends on (seam for the primary-store
migration to Postgres+RLS).

The in-memory `DataStore` exposes dict attributes (`store.documents`, etc.) that
callers filter in Python — incompatible with per-connection RLS. Migrating to
Postgres means moving callers (rbac/auth/tools) onto these org-scoped methods and
providing a `PostgresDataStore` whose reads run inside `org_transaction`. This
Protocol captures the contract so that refactor is mechanical and type-checked;
it needs a live database to verify end-to-end, so it is the next DB increment.
See plan §8.2 and §16.
"""

from __future__ import annotations

from typing import Iterable, Protocol

from app.domain import Chunk, Document, Section, User


class Repository(Protocol):
    # lookups
    def user_by_email(self, email: str) -> User | None: ...
    def document_by_contract_id(self, contract_id: str) -> Document | None: ...
    def sections_for_document(self, document_id: str) -> list[Section]: ...
    def section_by_number(self, document_id: str, section_number: str) -> Section | None: ...
    def chunks_for_documents(self, document_ids: Iterable[str]) -> list[Chunk]: ...

    # org-scoped collections (replace dict-attribute access under RLS)
    def documents_for_org(self, organization_id: str) -> list[Document]: ...
    def users_for_org(self, organization_id: str) -> list[User]: ...
