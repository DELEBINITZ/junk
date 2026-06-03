-- 0001_init: chat persistence + audit, with Row-Level-Security tenant isolation.
-- RLS is ENABLED and FORCED on every table; each policy requires the request's
-- org to match the row's org_id (set via SELECT set_config('app.organization_id', ...)).

CREATE TABLE IF NOT EXISTS chat_sessions (
    id            TEXT PRIMARY KEY,
    org_id        TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    title         TEXT NOT NULL DEFAULT 'New chat',
    summary       TEXT NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL DEFAULT 0,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sessions_org_user ON chat_sessions (org_id, user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    org_id      TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL DEFAULT '',
    citations   JSONB NOT NULL DEFAULT '[]'::jsonb,
    tool_calls  JSONB NOT NULL DEFAULT '[]'::jsonb,
    meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
    feedback    SMALLINT NOT NULL DEFAULT 0,          -- user rating: -1 down, 0 none, 1 up
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_org ON chat_messages (org_id, created_at DESC);
-- full-text search for cross-session recall
CREATE INDEX IF NOT EXISTS idx_messages_fts ON chat_messages USING gin (to_tsvector('english', content));

CREATE TABLE IF NOT EXISTS audit_log (
    id         BIGSERIAL PRIMARY KEY,
    org_id     TEXT NOT NULL,
    user_id    TEXT NOT NULL DEFAULT '',
    event      TEXT NOT NULL,
    payload    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_org ON audit_log (org_id, created_at DESC);

-- ---- Row-Level Security ----------------------------------------------------
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions FORCE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages FORCE ROW LEVEL SECURITY;
ALTER TABLE audit_log     ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log     FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS org_isolation ON chat_sessions;
CREATE POLICY org_isolation ON chat_sessions
    USING (org_id = current_setting('app.organization_id', true))
    WITH CHECK (org_id = current_setting('app.organization_id', true));

DROP POLICY IF EXISTS org_isolation ON chat_messages;
CREATE POLICY org_isolation ON chat_messages
    USING (org_id = current_setting('app.organization_id', true))
    WITH CHECK (org_id = current_setting('app.organization_id', true));

DROP POLICY IF EXISTS org_isolation ON audit_log;
CREATE POLICY org_isolation ON audit_log
    USING (org_id = current_setting('app.organization_id', true))
    WITH CHECK (org_id = current_setting('app.organization_id', true));
