"""PostgreSQL access with Row-Level-Security tenant isolation.

================================ MENTAL MODEL =============================
This module owns two things: a Postgres CONNECTION POOL (reuse a handful of open
connections instead of paying TCP+auth on every query) and — the important part —
the ONE primitive that enforces multi-tenant isolation at the database itself.

How RLS isolation works here, end to end:
  1. Each tenant-scoped table has a Row-Level Security (RLS) POLICY that says, in
     effect, "a row is visible only if its org_id equals the current org setting".
  2. That "current org setting" is a per-transaction GUC (Postgres's name for a
     runtime config variable), here ``app.organization_id``.
  3. ``org_transaction(org_id)`` opens a transaction and SETS that GUC for the
     transaction's lifetime, BEFORE running any of the caller's SQL.
  => Therefore every query inside that transaction is transparently filtered to
     the one tenant. A query that forgot a ``WHERE org_id=...`` STILL cannot see
     another org's rows — the database enforces it, not application code. This is
     defense-in-depth: the strongest place to put tenant isolation.

Critically, ``org_id`` is supplied by the caller from the VERIFIED token
(SecurityContext) — never from a request body — so it can't be spoofed. psycopg
is lazy-imported so the default in-memory store needs no Postgres driver at all.
===========================================================================
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.config import Settings


class PostgresDatabase:
    """Thin wrapper around an async psycopg connection pool that hands out
    tenant-scoped transactions. One instance is shared process-wide (see the
    module-level singleton below)."""

    def __init__(self, dsn: str, *, rls_setting: str = "app.organization_id",
                 min_size: int = 1, max_size: int = 10) -> None:
        if not dsn:
            raise ValueError("DATABASE_URL is required for store_backend=postgres")
        self.dsn = dsn
        self.rls_setting = rls_setting          # the GUC name the RLS policies read
        self._min, self._max = min_size, max_size   # pool sizing bounds
        self._pool = None                       # created lazily on first use (see open())

    async def open(self) -> None:
        """Create and open the connection pool once (idempotent). Called at startup
        or lazily on the first transaction. ``open=False`` then ``open(wait=True)``
        ensures the pool is fully warmed (connections established) before we use it."""
        if self._pool is not None:
            return
        from psycopg_pool import AsyncConnectionPool  # lazy import keeps the driver optional

        self._pool = AsyncConnectionPool(
            conninfo=self.dsn, min_size=self._min, max_size=self._max, open=False
        )
        await self._pool.open(wait=True)

    async def close(self) -> None:
        """Drain and close the pool on shutdown so connections are released cleanly."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def org_transaction(self, org_id: str) -> AsyncIterator:
        """Transaction scoped to one tenant. Sets the RLS GUC for its lifetime.

        This is the isolation primitive every PG-backed store call goes through.
        ``set_config(name, value, true)`` sets the GUC with ``is_local=true`` so it
        applies only to THIS transaction (and is rolled back when the transaction
        ends) — there's no risk of it leaking to the next caller that borrows the
        same pooled connection. The caller then runs its SQL against the yielded
        connection, fully scoped to ``org_id`` by the RLS policies."""
        if not org_id:                           # refuse to run un-scoped: no org => no transaction
            raise ValueError("org_transaction requires org_id")
        if self._pool is None:
            await self.open()
        async with self._pool.connection() as conn:    # borrow a pooled connection
            async with conn.transaction():              # BEGIN ... COMMIT/ROLLBACK
                await conn.execute(
                    "SELECT set_config(%s, %s, true)", (self.rls_setting, str(org_id))
                )                                       # arm RLS for this transaction
                yield conn

    @asynccontextmanager
    async def privileged_transaction(self) -> AsyncIterator:
        """Cross-tenant transaction for admin/audit reads. Use sparingly; the
        connecting role should still be RLS-bound unless it is a trusted admin.

        Deliberately does NOT set the org GUC, so it is NOT scoped to one tenant —
        that is its whole purpose (e.g. a platform-wide audit). Because it bypasses
        the per-tenant filter, it must only ever be used for genuinely cross-tenant
        admin work, never on the user-facing chat path."""
        if self._pool is None:
            await self.open()
        async with self._pool.connection() as conn:
            async with conn.transaction():
                yield conn


# Module-level singleton: one shared pool per process. Building a second pool would
# waste connections, so get_database memoizes the instance here.
_db: PostgresDatabase | None = None


def get_database(settings: Settings) -> PostgresDatabase:
    """Return the process-wide PostgresDatabase, constructing it (with pool sizing
    and the RLS GUC name from settings) on first call. Subsequent calls reuse it."""
    global _db
    if _db is None:
        _db = PostgresDatabase(
            settings.database_url, rls_setting=settings.rls_setting_name,
            min_size=settings.db_pool_min, max_size=settings.db_pool_max,
        )
    return _db


def reset_database() -> None:
    """Drop the singleton so the next get_database builds a fresh one. Used by
    tests (and reconfiguration) to avoid leaking a pool across instances."""
    global _db
    _db = None


__all__ = ["PostgresDatabase", "get_database", "reset_database"]
