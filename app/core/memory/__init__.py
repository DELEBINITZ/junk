"""Memory: chat persistence (Postgres) + the rolling conversation summary."""

from app.core.memory.conversations import ConversationStore, Message, Session
from app.core.memory.summarizer import RollingSummarizer


def build_conversation_store(settings) -> ConversationStore:
    # Postgres is the only chat/session store (no in-memory fallback).
    from app.core.db.postgres import get_database
    from app.core.memory.pg_conversations import PostgresConversationStore

    return PostgresConversationStore(get_database(settings), settings)


__all__ = [
    "Message",
    "Session",
    "ConversationStore",
    "RollingSummarizer",
    "build_conversation_store",
]
