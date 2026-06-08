from contextlib import asynccontextmanager
from typing import AsyncIterator

import psycopg
from psycopg_pool import AsyncConnectionPool

from security_intel.config import Settings


class Database:
    """Async Postgres connection pool with explicit org_id filtering (no RLS)."""

    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    @classmethod
    async def connect(cls, settings: Settings) -> "Database":
        pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=2,
            max_size=10,
            open=False,
        )
        await pool.open()
        return cls(pool)

    async def close(self):
        await self._pool.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[psycopg.AsyncCursor]:
        """Get a cursor within a transaction. Org filtering is done in SQL WHERE clauses."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                yield cur
            await conn.commit()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[psycopg.AsyncConnection]:
        """Raw connection for migrations and admin ops."""
        async with self._pool.connection() as conn:
            yield conn
