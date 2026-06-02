-- 0001_chat_audit: durable chat persistence + append-only audit, org-isolated
-- via row-level security. Complements the existing core schema (organizations,
-- users, documents, ...). RLS GUC: app.organization_id (set per transaction by
-- app.core.db.postgres.org_transaction). See plan §9 and §8.

CREATE TABLE IF NOT EXISTS sessions (
  id uuid PRIMARY KEY,
  organization_id text NOT NULL,
  user_id text NOT NULL,
  title text NOT NULL DEFAULT 'New chat',
  rolling_summary text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  soft_deleted_at timestamptz NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id uuid PRIMARY KEY,
  session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  organization_id text NOT NULL,
  user_id text NOT NULL,
  role text NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
  content text NOT NULL,
  citations jsonb NOT NULL DEFAULT '[]',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_state (
  session_id uuid PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  organization_id text NOT NULL,
  graph_state jsonb NOT NULL DEFAULT '{}',
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
  id uuid PRIMARY KEY,
  organization_id text,
  user_id text,
  action text NOT NULL,
  resource_type text,
  resource_id text,
  outcome text,
  details jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

ALTER TABLE sessions FORCE ROW LEVEL SECURITY;
ALTER TABLE messages FORCE ROW LEVEL SECURITY;
ALTER TABLE chat_state FORCE ROW LEVEL SECURITY;
ALTER TABLE audit_log FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_sessions_select ON sessions
  USING (organization_id = current_setting('app.organization_id', true));
CREATE POLICY tenant_sessions_write ON sessions
  WITH CHECK (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_messages_select ON messages
  USING (organization_id = current_setting('app.organization_id', true));
CREATE POLICY tenant_messages_write ON messages
  WITH CHECK (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_chat_state_select ON chat_state
  USING (organization_id = current_setting('app.organization_id', true));
CREATE POLICY tenant_chat_state_write ON chat_state
  WITH CHECK (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_audit_log_select ON audit_log
  USING (organization_id = current_setting('app.organization_id', true));
CREATE POLICY tenant_audit_log_write ON audit_log
  WITH CHECK (organization_id = current_setting('app.organization_id', true));

CREATE INDEX IF NOT EXISTS messages_session_created_idx ON messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS messages_user_idx ON messages (organization_id, user_id);
CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions (organization_id, user_id);
CREATE INDEX IF NOT EXISTS audit_log_org_created_idx ON audit_log (organization_id, created_at);
