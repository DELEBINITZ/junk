"""Memory: chat persistence, rolling summary, long-term KG seam."""

from app.core.memory.conversations import (
    ConversationStore,
    InMemoryConversationStore,
    Message,
    Session,
)
from app.core.memory.kg import KnowledgeGraph, NoOpKnowledgeGraph, build_kg
from app.core.memory.summarizer import RollingSummarizer


def build_conversation_store(settings) -> ConversationStore:
    if settings.store_backend == "postgres":
        from app.core.db.postgres import get_database
        from app.core.memory.pg_conversations import PostgresConversationStore

        return PostgresConversationStore(get_database(settings), settings)
    return InMemoryConversationStore()


__all__ = [
    "Message",
    "Session",
    "ConversationStore",
    "InMemoryConversationStore",
    "RollingSummarizer",
    "KnowledgeGraph",
    "NoOpKnowledgeGraph",
    "build_kg",
    "build_conversation_store",
]
