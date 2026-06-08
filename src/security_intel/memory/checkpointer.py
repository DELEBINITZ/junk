from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from security_intel.config import Settings


async def get_checkpointer(settings: Settings) -> AsyncPostgresSaver:
    """Create PostgreSQL-backed checkpointer for durable graph state."""
    checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url)
    await checkpointer.setup()
    return checkpointer
