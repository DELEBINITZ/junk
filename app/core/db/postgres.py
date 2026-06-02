"""PostgreSQL access with the org-isolation backbone.

`org_transaction(org_id)` opens a transaction and sets the RLS GUC
(`app.organization_id`) LOCAL to that transaction. Every query inside is then
constrained by the row-level-security policies in the migrations, so a forgotten
WHERE clause cannot leak another org's rows. This is the security primitive the
rest of the persistence layer is built on (plan §8.2).

`psycopg` (the `prod` extra) is imported lazily, so importing this module — and
running the default in-memory path/tests — never requires the driver. A `connect`
callable can be injected for unit tests (no real database needed).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from app.config import settings


logger = logging.getLogger(__name__)

ConnectFn = Callable[[str], Any]


class PostgresUnavailable(RuntimeError):
    """Raised when the psycopg driver is missing or the database is unreachable."""


class PostgresDatabase:
    def __init__(
        self,
        dsn: str | None = None,
        rls_setting: str | None = None,
        connect: ConnectFn | None = None,
    ):
        self.dsn = dsn or settings.database_url
        self.rls_setting = rls_setting or settings.rls_setting_name
        self._connect = connect  # injectable for tests

    def _open(self) -> Any:
        if self._connect is not None:
            return self._connect(self.dsn)
        try:
            import psycopg
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise PostgresUnavailable(
                "psycopg is not installed. Install the 'prod' extra to use the "
                "postgres store backend."
            ) from exc
        return psycopg.connect(self.dsn)

    @contextmanager
    def org_transaction(self, organization_id: str) -> Iterator[Any]:
        """Open an org-scoped transaction. The RLS GUC is set LOCAL, so it resets
        automatically when the transaction ends. Returns the live connection for
        queries; commits on success, always closes."""

        conn = self._open()
        try:
            with conn.transaction():
                conn.execute(
                    "SELECT set_config(%s, %s, true)",
                    (self.rls_setting, str(organization_id)),
                )
                yield conn
        finally:
            conn.close()

    @contextmanager
    def privileged_transaction(self) -> Iterator[Any]:
        """Open a transaction with NO org GUC set. Intended to be run as a role
        with BYPASSRLS (or the table owner) for the two operations that legitimately
        cross orgs: authentication user-lookup (no org context exists yet at login)
        and admin seeding/migrations. The request/agent path NEVER uses this — it
        uses org_transaction so RLS applies. See plan §8.2."""

        conn = self._open()
        try:
            with conn.transaction():
                yield conn
        finally:
            conn.close()

    def health_check(self) -> bool:
        try:
            conn = self._open()
        except PostgresUnavailable:
            return False
        try:
            with conn.transaction():
                conn.execute("SELECT 1")
            return True
        except Exception:  # pragma: no cover - env-dependent
            return False
        finally:
            try:
                conn.close()
            except Exception:  # pragma: no cover
                pass


_database: PostgresDatabase | None = None


def get_database() -> PostgresDatabase:
    global _database
    if _database is None:
        _database = PostgresDatabase()
    return _database


def reset_database() -> None:
    global _database
    _database = None
