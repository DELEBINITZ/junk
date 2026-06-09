from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from security_intel.config import Settings


async def get_checkpointer(settings: Settings) -> tuple[AsyncPostgresSaver, AsyncConnectionPool]:
    """Create PostgreSQL-backed checkpointer for durable graph state.

    Returns (checkpointer, pool) — caller must close pool on shutdown.
    """
    # autocommit=True is REQUIRED: AsyncPostgresSaver.setup() runs
    # CREATE INDEX CONCURRENTLY, which Postgres forbids inside a transaction.
    # prepare_threshold=0 avoids prepared-statement clashes across pooled conns.
    pool = AsyncConnectionPool(
        settings.database_url,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    return checkpointer, pool
