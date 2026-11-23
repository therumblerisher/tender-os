-- Tender OS schema (2 tables only: tenders + spend_ledger)
-- Vector dimension 1024 matches PR0 winner: YTL ILMU bge-m3 (lowercased).
-- 'simple' FTS configuration tokenizes Bahasa Melayu, English, and 6-digit MOF codes
-- without English stemmer distortion.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tenders (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    reference TEXT NOT NULL,
    title TEXT NOT NULL,
    agency TEXT NOT NULL,
    category TEXT,
    location TEXT,
    closing_at TEXT,
    value_myr TEXT,
    eligibility TEXT,
    url TEXT NOT NULL,
    published_at TEXT,
    search_text TEXT NOT NULL,
    search_tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(reference, '') || ' ' || coalesce(eligibility, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(agency, '') || ' ' || coalesce(category, '') || ' ' || coalesce(location, '')), 'C')
    ) STORED,
    embedding vector(1024),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tenders_tsv
    ON tenders USING GIN (search_tsv);

CREATE INDEX IF NOT EXISTS idx_tenders_embedding_hnsw
    ON tenders USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_tenders_source
    ON tenders (source);

CREATE INDEX IF NOT EXISTS idx_tenders_closing_at
    ON tenders (closing_at);

-- Daily MYT budget ledger. reserved_micro_rm tracks in-flight holds (returns to 0 when idle);
-- settled_micro_rm tracks finalized spend for the day. Active budget check: reserved + settled <= cap.
CREATE TABLE IF NOT EXISTS spend_ledger (
    day_myt DATE PRIMARY KEY,
    reserved_micro_rm BIGINT NOT NULL DEFAULT 0 CHECK (reserved_micro_rm >= 0),
    settled_micro_rm BIGINT NOT NULL DEFAULT 0 CHECK (settled_micro_rm >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
