"""PostgreSQL access with Row-Level-Security tenant isolation.

``org_transaction(org_id)`` is THE isolation primitive: it opens a transaction
and sets the ``app.organization_id`` GUC, which every table's RLS policy checks —
so even a buggy query cannot read another tenant's rows. The conversation store
(and any future PG-backed store) only ever touches rows through this. psycopg is
lazy-imported so the default in-memory path needs no driver.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.config import Settings


class PostgresDatabase:
    def __init__(self, dsn: str, *, rls_setting: str = "app.organization_id",
                 min_size: int = 1, max_size: int = 10) -> None:
        if not dsn:
            raise ValueError("DATABASE_URL is required for store_backend=postgres")
        self.dsn = dsn
        self.rls_setting = rls_setting
        self._min, self._max = min_size, max_size
        self._pool = None

    async def open(self) -> None:
        if self._pool is not None:
            return
        from psycopg_pool import AsyncConnectionPool  # lazy

        self._pool = AsyncConnectionPool(
            conninfo=self.dsn, min_size=self._min, max_size=self._max, open=False
        )
        await self._pool.open(wait=True)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def org_transaction(self, org_id: str) -> AsyncIterator:
        """Transaction scoped to one tenant. Sets the RLS GUC for its lifetime."""
        if not org_id:
            raise ValueError("org_transaction requires org_id")
        if self._pool is None:
            await self.open()
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config(%s, %s, true)", (self.rls_setting, str(org_id))
                )
                yield conn

    @asynccontextmanager
    async def privileged_transaction(self) -> AsyncIterator:
        """Cross-tenant transaction for admin/audit reads. Use sparingly; the
        connecting role should still be RLS-bound unless it is a trusted admin."""
        if self._pool is None:
            await self.open()
        async with self._pool.connection() as conn:
            async with conn.transaction():
                yield conn


_db: PostgresDatabase | None = None


def get_database(settings: Settings) -> PostgresDatabase:
    global _db
    if _db is None:
        _db = PostgresDatabase(
            settings.database_url, rls_setting=settings.rls_setting_name,
            min_size=settings.db_pool_min, max_size=settings.db_pool_max,
        )
    return _db


def reset_database() -> None:
    global _db
    _db = None


__all__ = ["PostgresDatabase", "get_database", "reset_database"]
