from contextlib import asynccontextmanager
from typing import AsyncIterator

import psycopg
from psycopg_pool import AsyncConnectionPool

from security_intel.config import Settings


class Database:
    """Async Postgres connection pool with RLS enforcement."""

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
    async def org_transaction(self, org_id: str) -> AsyncIterator[psycopg.AsyncCursor]:
        """Get a cursor with RLS set to the given org_id.

        All queries within this context are automatically scoped to the org.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SET LOCAL app.organization_id = %s", (org_id,)
                )
                yield cur
            await conn.commit()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[psycopg.AsyncConnection]:
        """Raw connection without RLS (for migrations, admin ops)."""
        async with self._pool.connection() as conn:
            yield conn
