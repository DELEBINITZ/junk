"""Database schema initialization. Run once on first deploy."""

SCHEMA_SQL = """
-- Chat sessions (conversations)
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL DEFAULT 0,
    summarized_upto INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_org_user
    ON chat_sessions(org_id, user_id, updated_at DESC);

-- Chat messages (individual turns)
CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    org_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL DEFAULT '',
    citations JSONB NOT NULL DEFAULT '[]',
    tool_calls JSONB NOT NULL DEFAULT '[]',
    meta JSONB NOT NULL DEFAULT '{}',
    feedback SMALLINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON chat_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_org
    ON chat_messages(org_id, created_at DESC);

-- Row-Level Security
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS sessions_org_isolation ON chat_sessions;
CREATE POLICY sessions_org_isolation ON chat_sessions
    USING (org_id = current_setting('app.organization_id', true));

ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS messages_org_isolation ON chat_messages;
CREATE POLICY messages_org_isolation ON chat_messages
    USING (org_id = current_setting('app.organization_id', true));
"""


async def run_migrations(database) -> None:
    """Apply schema. Idempotent (IF NOT EXISTS everywhere)."""
    async with database.connection() as conn:
        await conn.execute(SCHEMA_SQL)
        await conn.commit()
    print("Database migrations applied.")
