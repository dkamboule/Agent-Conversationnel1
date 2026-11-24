-- ============================================================================
-- DGECC — Schéma PostgreSQL + pgvector (Modèle Physique de Données)
-- Direction Générale des Examens, Concours et de la Certification
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "unaccent";

-- Séparation logique par schémas
CREATE SCHEMA IF NOT EXISTS admin_security;
CREATE SCHEMA IF NOT EXISTS corpus;
CREATE SCHEMA IF NOT EXISTS indexation;
CREATE SCHEMA IF NOT EXISTS conversational;

-- ----------------------------------------------------------------------------
-- 1. DONNÉES ADMINISTRATIVES & SÉCURITÉ (admin_security)
-- ----------------------------------------------------------------------------
CREATE TABLE admin_security.roles (
    id              SERIAL PRIMARY KEY,
    code            VARCHAR(32) UNIQUE NOT NULL CHECK (code IN ('ADMIN', 'VALIDATOR', 'EDITOR', 'USER')),
    label           VARCHAR(120) NOT NULL,
    permissions     JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE admin_security.users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    full_name       VARCHAR(160) NOT NULL,
    password_hash   TEXT NOT NULL,
    salt            TEXT NOT NULL,
    role_code       VARCHAR(32) NOT NULL REFERENCES admin_security.roles(code),
    service         VARCHAR(160),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE admin_security.activity_logs (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID REFERENCES admin_security.users(id) ON DELETE SET NULL,
    user_email      VARCHAR(255),
    action          VARCHAR(100) NOT NULL,
    resource_type   VARCHAR(64) NOT NULL,
    resource_id     VARCHAR(128),
    details         JSONB NOT NULL DEFAULT '{}'::jsonb,
    ip_address      INET,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_activity_logs_created_at ON admin_security.activity_logs(created_at DESC);
CREATE INDEX idx_activity_logs_action ON admin_security.activity_logs(action);

-- ----------------------------------------------------------------------------
-- 2. DONNÉES DOCUMENTAIRES (corpus)
-- ----------------------------------------------------------------------------
CREATE TABLE corpus.documents (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title               VARCHAR(300) NOT NULL,
    doc_type            VARCHAR(80) NOT NULL,
    category            VARCHAR(120) NOT NULL,
    service_name        VARCHAR(180) NOT NULL,
    reference_code      VARCHAR(80) NOT NULL,
    published_date      DATE NOT NULL,
    valid_until         DATE,
    current_version     INTEGER NOT NULL DEFAULT 1,
    status              VARCHAR(32) NOT NULL DEFAULT 'BROUILLON'
                        CHECK (status IN ('BROUILLON', 'EN_ATTENTE', 'VALIDE', 'PUBLIE', 'ARCHIVE', 'REJETE')),
    file_name           VARCHAR(255) NOT NULL,
    file_format         VARCHAR(16) NOT NULL CHECK (file_format IN ('PDF', 'DOCX', 'HTML', 'TXT')),
    uploaded_by         UUID REFERENCES admin_security.users(id) ON DELETE SET NULL,
    validated_by        UUID REFERENCES admin_security.users(id) ON DELETE SET NULL,
    rejection_reason    TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_documents_status ON corpus.documents(status);
CREATE INDEX idx_documents_category ON corpus.documents(category);
CREATE INDEX idx_documents_reference ON corpus.documents(reference_code);

CREATE TABLE corpus.document_versions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id         UUID NOT NULL REFERENCES corpus.documents(id) ON DELETE CASCADE,
    version_number      INTEGER NOT NULL,
    raw_text            TEXT NOT NULL,
    normalized_text     TEXT NOT NULL,
    chunk_size          INTEGER NOT NULL DEFAULT 800,
    chunk_overlap       INTEGER NOT NULL DEFAULT 150,
    created_by          UUID REFERENCES admin_security.users(id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, version_number)
);

CREATE TABLE corpus.document_metadata (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES corpus.documents(id) ON DELETE CASCADE,
    meta_key        VARCHAR(100) NOT NULL,
    meta_value      TEXT NOT NULL
);

-- ----------------------------------------------------------------------------
-- 3. DONNÉES D'INDEXATION (indexation : BM25 + pgvector)
-- ----------------------------------------------------------------------------
CREATE TABLE indexation.document_chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES corpus.documents(id) ON DELETE CASCADE,
    version_number  INTEGER NOT NULL,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    tsv_french      TSVECTOR GENERATED ALWAYS AS (to_tsvector('french', unaccent(content))) STORED,
    token_count     INTEGER NOT NULL,
    char_count      INTEGER NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, version_number, chunk_index)
);
CREATE INDEX idx_chunks_tsv_french ON indexation.document_chunks USING GIN (tsv_french);
CREATE INDEX idx_chunks_doc_ver ON indexation.document_chunks(document_id, version_number);

CREATE TABLE indexation.embeddings (
    chunk_id        UUID PRIMARY KEY REFERENCES indexation.document_chunks(id) ON DELETE CASCADE,
    embedding       vector(768) NOT NULL,
    model_name      VARCHAR(100) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_embeddings_hnsw ON indexation.embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE indexation.indexation_history (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id         UUID NOT NULL REFERENCES corpus.documents(id) ON DELETE CASCADE,
    version_number      INTEGER NOT NULL,
    chunk_count         INTEGER NOT NULL,
    embedding_model     VARCHAR(100) NOT NULL,
    bm25_terms_count    INTEGER NOT NULL,
    duration_ms         INTEGER NOT NULL,
    status              VARCHAR(32) NOT NULL CHECK (status IN ('SUCCES', 'ECHEC')),
    triggered_by        UUID REFERENCES admin_security.users(id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- 4. DONNÉES CONVERSATIONNELLES (conversational)
-- ----------------------------------------------------------------------------
CREATE TABLE conversational.conversations (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID REFERENCES admin_security.users(id) ON DELETE CASCADE,
    guest_session_id    VARCHAR(128),
    title               VARCHAR(255) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_conversations_user ON conversational.conversations(user_id, updated_at DESC);

CREATE TABLE conversational.messages (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id     UUID NOT NULL REFERENCES conversational.conversations(id) ON DELETE CASCADE,
    role                VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
    content             TEXT NOT NULL,
    nlp_analysis        JSONB,
    guardrails_report   JSONB,
    metrics             JSONB,
    feedback            VARCHAR(16) CHECK (feedback IN ('positive', 'negative')),
    feedback_comment    TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE conversational.message_sources (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id      UUID NOT NULL REFERENCES conversational.messages(id) ON DELETE CASCADE,
    document_id     UUID NOT NULL REFERENCES corpus.documents(id) ON DELETE CASCADE,
    chunk_id        UUID NOT NULL REFERENCES indexation.document_chunks(id) ON DELETE CASCADE,
    bm25_rank       INTEGER,
    vector_rank     INTEGER,
    rrf_score       DOUBLE PRECISION NOT NULL,
    rerank_score    DOUBLE PRECISION NOT NULL
);
