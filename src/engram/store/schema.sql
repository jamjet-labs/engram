-- Engram v2 SQLite schema (Phase 1)
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
INSERT OR IGNORE INTO schema_version (version) VALUES (1);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    org_id      TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
    content     TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(org_id, user_id, session_id, timestamp);

CREATE TABLE IF NOT EXISTS facts (
    id              TEXT PRIMARY KEY,
    org_id          TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    text            TEXT NOT NULL,
    valid_from      TEXT NOT NULL,
    invalid_at      TEXT,
    confidence      REAL NOT NULL DEFAULT 1.0,
    category        TEXT,
    polarity        TEXT NOT NULL DEFAULT 'affirmative',
    tier            TEXT NOT NULL DEFAULT 'working',
    event_date      TEXT,
    mention_date    TEXT,
    source_event_id TEXT,
    source_message_id TEXT,
    source_span_start INTEGER,
    source_span_end   INTEGER,
    supersedes      TEXT,
    superseded_by   TEXT,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   TEXT,
    metadata        TEXT NOT NULL DEFAULT '{}',
    session_id      TEXT,
    FOREIGN KEY (source_message_id) REFERENCES messages(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_scope
    ON facts(org_id, user_id, valid_from);
CREATE INDEX IF NOT EXISTS idx_facts_session
    ON facts(org_id, user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_facts_event_date
    ON facts(org_id, user_id, event_date);
CREATE INDEX IF NOT EXISTS idx_facts_category
    ON facts(org_id, user_id, category);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    org_id      TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    aliases     TEXT NOT NULL DEFAULT '[]',
    metadata    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_entities_name
    ON entities(org_id, user_id, name);

CREATE TABLE IF NOT EXISTS fact_entity_refs (
    fact_id     TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    PRIMARY KEY (fact_id, entity_id),
    FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS relationships (
    id          TEXT PRIMARY KEY,
    org_id      TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    source      TEXT NOT NULL,
    relation    TEXT NOT NULL,
    target      TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_relationships_source
    ON relationships(org_id, user_id, source);
CREATE INDEX IF NOT EXISTS idx_relationships_target
    ON relationships(org_id, user_id, target);

-- ── Events (Phase 11: SVO event calendar) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS events (
    id                  TEXT PRIMARY KEY,
    org_id              TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    subject_canonical   TEXT NOT NULL,
    verb                TEXT NOT NULL,
    object_canonical    TEXT NOT NULL,
    time_start          TEXT NOT NULL,
    time_end            TEXT,
    confidence          REAL NOT NULL DEFAULT 1.0,
    aliases             TEXT NOT NULL DEFAULT '[]',
    source_fact_ids     TEXT NOT NULL DEFAULT '[]',
    metadata            TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_scope ON events(org_id, user_id, time_start);
CREATE INDEX IF NOT EXISTS idx_events_subject ON events(org_id, user_id, subject_canonical);
CREATE INDEX IF NOT EXISTS idx_events_object ON events(org_id, user_id, object_canonical);
CREATE INDEX IF NOT EXISTS idx_events_verb ON events(org_id, user_id, verb);

-- FTS5 over canonical SVO + aliases for natural-language event search.
-- Contentless FTS5 (no content= linkage) — we maintain text via triggers.
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    text,
    org_id UNINDEXED,
    user_id UNINDEXED,
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS events_fts_insert AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, text, org_id, user_id)
    VALUES (
        new.rowid,
        new.subject_canonical || ' ' || new.verb || ' ' || new.object_canonical || ' ' || new.aliases,
        new.org_id,
        new.user_id
    );
END;
CREATE TRIGGER IF NOT EXISTS events_fts_delete AFTER DELETE ON events BEGIN
    DELETE FROM events_fts WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS events_fts_update AFTER UPDATE ON events BEGIN
    DELETE FROM events_fts WHERE rowid = old.rowid;
    INSERT INTO events_fts(rowid, text, org_id, user_id)
    VALUES (
        new.rowid,
        new.subject_canonical || ' ' || new.verb || ' ' || new.object_canonical || ' ' || new.aliases,
        new.org_id,
        new.user_id
    );
END;

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    text,
    org_id UNINDEXED,
    user_id UNINDEXED,
    content='facts',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS facts_fts_insert AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, text, org_id, user_id)
    VALUES (new.rowid, new.text, new.org_id, new.user_id);
END;
CREATE TRIGGER IF NOT EXISTS facts_fts_delete AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text, org_id, user_id)
    VALUES ('delete', old.rowid, old.text, old.org_id, old.user_id);
END;
CREATE TRIGGER IF NOT EXISTS facts_fts_update AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text, org_id, user_id)
    VALUES ('delete', old.rowid, old.text, old.org_id, old.user_id);
    INSERT INTO facts_fts(rowid, text, org_id, user_id)
    VALUES (new.rowid, new.text, new.org_id, new.user_id);
END;
