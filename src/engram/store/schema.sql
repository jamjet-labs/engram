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

-- ── Governance (0.3): provenance, revocation, erasure ──────────────────────
-- Side tables rather than new columns on `facts`, so databases created by
-- 0.1/0.2 pick them up on open without a migration.

CREATE TABLE IF NOT EXISTS fact_meta (
    fact_id       TEXT PRIMARY KEY NOT NULL,
    org_id        TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    writer        TEXT,
    source        TEXT NOT NULL,
    recorded_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_fact_meta_hash
    ON fact_meta(org_id, user_id, content_hash);

CREATE TABLE IF NOT EXISTS fact_derivations (
    fact_id         TEXT NOT NULL,
    source_fact_id  TEXT NOT NULL,
    PRIMARY KEY (fact_id, source_fact_id)
);
CREATE INDEX IF NOT EXISTS idx_fact_derivations_source
    ON fact_derivations(source_fact_id);

CREATE TABLE IF NOT EXISTS revocations (
    id               TEXT PRIMARY KEY NOT NULL,
    org_id           TEXT NOT NULL,
    user_id          TEXT NOT NULL,
    fact_id          TEXT NOT NULL,
    content_hash     TEXT,
    -- Normalized revoked text, used to find later writes that quote it.
    -- NULL when the text normalizes to nothing, and after erasure.
    content_norm     TEXT,
    reason           TEXT NOT NULL,
    actor            TEXT NOT NULL,
    revoked_at       TEXT NOT NULL,
    replaced_by      TEXT,
    reinstated_at    TEXT,
    reinstated_by    TEXT,
    reinstate_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_revocations_fact ON revocations(fact_id);
CREATE INDEX IF NOT EXISTS idx_revocations_hash
    ON revocations(org_id, user_id, content_hash);

-- Items (facts, transcript turns, events) whose text quotes a revoked text.
-- Computed when the revocation is written and when the item is written, so
-- the view below stays an indexed lookup instead of a text scan per query.
CREATE TABLE IF NOT EXISTS content_matches (
    kind           TEXT NOT NULL CHECK (kind IN ('fact','message','event')),
    item_id        TEXT NOT NULL,
    revocation_id  TEXT NOT NULL,
    PRIMARY KEY (kind, item_id, revocation_id)
);
CREATE INDEX IF NOT EXISTS idx_content_matches_rev ON content_matches(revocation_id);

CREATE TABLE IF NOT EXISTS erasures (
    id          TEXT PRIMARY KEY NOT NULL,
    org_id      TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    fact_ids    TEXT NOT NULL,
    actor       TEXT NOT NULL,
    reason      TEXT NOT NULL,
    erased_at   TEXT NOT NULL,
    artifacts   TEXT NOT NULL DEFAULT '{}',
    guarantees  TEXT NOT NULL DEFAULT '{}'
);

-- The single visibility decision. Recreated on every open so a newer library
-- version always runs its own definition against an older database file.
DROP VIEW IF EXISTS live_messages;
DROP VIEW IF EXISTS live_events;
DROP VIEW IF EXISTS live_facts;
DROP VIEW IF EXISTS hidden_facts;

CREATE VIEW hidden_facts AS
WITH RECURSIVE hidden(id) AS (
    SELECT r.fact_id
      FROM revocations r
     WHERE r.reinstated_at IS NULL
    UNION
    SELECT m.fact_id
      FROM fact_meta m
      JOIN revocations r
        ON r.content_hash = m.content_hash
       AND r.org_id = m.org_id
       AND r.user_id = m.user_id
     WHERE r.reinstated_at IS NULL
    UNION
    SELECT cm.item_id
      FROM content_matches cm
      JOIN revocations r ON r.id = cm.revocation_id
     WHERE cm.kind = 'fact' AND r.reinstated_at IS NULL
    UNION
    SELECT d.fact_id
      FROM fact_derivations d
      JOIN hidden h ON d.source_fact_id = h.id
)
SELECT id FROM hidden;

CREATE VIEW live_facts AS
SELECT facts.rowid AS fact_rowid, facts.*
  FROM facts
 WHERE facts.id NOT IN (SELECT id FROM hidden_facts);

-- An event is withheld when any fact it was derived from is hidden.
-- Events with no source_fact_ids cannot be linked and stay visible.
CREATE VIEW live_events AS
SELECT events.rowid AS event_rowid, events.*
  FROM events
 WHERE NOT EXISTS (
        SELECT 1 FROM json_each(events.source_fact_ids) j
         WHERE j.value IN (SELECT id FROM hidden_facts)
       )
   AND NOT EXISTS (
        SELECT 1 FROM content_matches cm
          JOIN revocations r ON r.id = cm.revocation_id
         WHERE cm.kind = 'event' AND cm.item_id = events.id
           AND r.reinstated_at IS NULL
       );

-- A transcript turn is withheld when (a) a hidden fact points at it via
-- source_message_id, (b) its text quotes a revoked text (content_matches), or
-- (c) an operator revoked a fact by id that sits in the turn's session with no
-- message link, since there is no way to tell which turn carried it. Rule (c)
-- keys on the revocation itself, not on facts hidden by content or
-- derivation, so a plain write of revoked text cannot withhold a session.
CREATE VIEW live_messages AS
SELECT messages.*
  FROM messages
 WHERE messages.id NOT IN (
        SELECT f.source_message_id FROM facts f
         WHERE f.source_message_id IS NOT NULL
           AND f.id IN (SELECT id FROM hidden_facts)
       )
   AND NOT EXISTS (
        SELECT 1 FROM content_matches cm
          JOIN revocations r ON r.id = cm.revocation_id
         WHERE cm.kind = 'message' AND cm.item_id = messages.id
           AND r.reinstated_at IS NULL
       )
   AND NOT EXISTS (
        SELECT 1 FROM revocations r
          JOIN facts f ON f.id = r.fact_id
         WHERE r.reinstated_at IS NULL
           AND f.source_message_id IS NULL
           AND f.session_id = messages.session_id
           AND f.org_id = messages.org_id
           AND f.user_id = messages.user_id
       );
