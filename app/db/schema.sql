CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE organizations (
  id text PRIMARY KEY,
  name text NOT NULL
);

CREATE TABLE users (
  id text PRIMARY KEY,
  organization_id text NOT NULL REFERENCES organizations(id),
  email text UNIQUE NOT NULL,
  name text NOT NULL,
  role text NOT NULL CHECK (role IN ('admin', 'analyst', 'viewer')),
  password_hash text NOT NULL,
  is_active boolean DEFAULT true
);

CREATE TABLE documents (
  id uuid PRIMARY KEY,
  organization_id text NOT NULL REFERENCES organizations(id),
  contract_id text UNIQUE,
  title text NOT NULL,
  filename text,
  uploaded_by text NOT NULL REFERENCES users(id),
  tags text[] DEFAULT '{}',
  raw_text text,
  redacted_text text,
  metadata jsonb DEFAULT '{}',
  expiration_date date,
  notice_period_days int,
  contract_value_numeric numeric,
  deleted_at timestamp NULL,
  created_at timestamp DEFAULT now()
);

CREATE TABLE document_shares (
  document_id uuid REFERENCES documents(id),
  user_id text REFERENCES users(id),
  access_level text CHECK (access_level IN ('read', 'query', 'edit')),
  PRIMARY KEY (document_id, user_id)
);

CREATE TABLE sections (
  id uuid PRIMARY KEY,
  document_id uuid REFERENCES documents(id),
  organization_id text NOT NULL,
  section_number text,
  section_title text,
  page_number int,
  line_start int,
  line_end int,
  text text NOT NULL
);

CREATE TABLE chunks (
  id uuid PRIMARY KEY,
  document_id uuid REFERENCES documents(id),
  organization_id text NOT NULL,
  section_id uuid REFERENCES sections(id),
  chunk_index int,
  text text NOT NULL,
  embedding vector(384),
  metadata jsonb DEFAULT '{}'
);

CREATE TABLE guardrail_configs (
  organization_id text PRIMARY KEY REFERENCES organizations(id),
  hallucination_confidence_threshold float DEFAULT 0.7,
  pii_redaction_enabled boolean DEFAULT true,
  blocked_keywords text[] DEFAULT '{}',
  require_citations boolean DEFAULT true,
  toxicity_threshold float DEFAULT 0.8
);

CREATE TABLE reports (
  id uuid PRIMARY KEY,
  organization_id text NOT NULL,
  created_by text NOT NULL REFERENCES users(id),
  title text,
  query text,
  result jsonb,
  created_at timestamp DEFAULT now()
);

CREATE TABLE audit_events (
  id uuid PRIMARY KEY,
  organization_id text,
  user_id text,
  action text,
  resource_type text,
  resource_id text,
  outcome text,
  details jsonb,
  created_at timestamp DEFAULT now()
);

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_shares ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE sections ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;

ALTER TABLE documents FORCE ROW LEVEL SECURITY;
ALTER TABLE document_shares FORCE ROW LEVEL SECURITY;
ALTER TABLE users FORCE ROW LEVEL SECURITY;
ALTER TABLE sections FORCE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
ALTER TABLE reports FORCE ROW LEVEL SECURITY;
ALTER TABLE audit_events FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_documents_select ON documents
  USING (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_documents_write ON documents
  WITH CHECK (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_users_select ON users
  USING (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_users_write ON users
  WITH CHECK (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_document_shares_select ON document_shares
  USING (
    EXISTS (
      SELECT 1 FROM documents d
      WHERE d.id = document_id
        AND d.organization_id = current_setting('app.organization_id', true)
    )
  );

CREATE POLICY tenant_document_shares_write ON document_shares
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM documents d
      WHERE d.id = document_id
        AND d.organization_id = current_setting('app.organization_id', true)
    )
  );

CREATE POLICY tenant_sections_select ON sections
  USING (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_sections_write ON sections
  WITH CHECK (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_chunks_select ON chunks
  USING (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_chunks_write ON chunks
  WITH CHECK (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_reports_select ON reports
  USING (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_reports_write ON reports
  WITH CHECK (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_audit_events_select ON audit_events
  USING (organization_id = current_setting('app.organization_id', true));

CREATE POLICY tenant_audit_events_write ON audit_events
  WITH CHECK (organization_id = current_setting('app.organization_id', true));

CREATE INDEX chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX documents_org_expiration_idx ON documents (organization_id, expiration_date);
CREATE INDEX chunks_org_doc_idx ON chunks (organization_id, document_id);
