-- Grimoire initial schema. See plan §5.
-- sqlite-vec extension must be loaded on the connection before running this.

PRAGMA foreign_keys = ON;

CREATE TABLE items (
    id INTEGER PRIMARY KEY,
    item_type TEXT NOT NULL CHECK(item_type IN
        ('paper','book','chapter','report','thesis','preprint','standard','patent','other')),
    title TEXT NOT NULL,
    abstract TEXT,
    publication_year INTEGER,
    doi TEXT,
    arxiv_id TEXT,
    isbn TEXT,
    venue TEXT,
    volume TEXT,
    issue TEXT,
    pages TEXT,
    series TEXT,
    series_number TEXT,
    edition TEXT,
    language TEXT DEFAULT 'en',
    content_hash TEXT,
    file_path TEXT,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata_source TEXT,
    metadata_confidence REAL,
    metadata_json TEXT,
    notes TEXT
);

CREATE UNIQUE INDEX idx_items_doi ON items(doi) WHERE doi IS NOT NULL;
CREATE UNIQUE INDEX idx_items_arxiv ON items(arxiv_id) WHERE arxiv_id IS NOT NULL;
CREATE UNIQUE INDEX idx_items_hash ON items(content_hash) WHERE content_hash IS NOT NULL;

CREATE TABLE authors (
    id INTEGER PRIMARY KEY,
    family_name TEXT NOT NULL,
    given_name TEXT,
    orcid TEXT UNIQUE,
    normalized_key TEXT NOT NULL,
    UNIQUE(normalized_key, orcid)
);
CREATE INDEX idx_authors_normkey ON authors(normalized_key);

CREATE TABLE item_authors (
    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    author_id INTEGER REFERENCES authors(id),
    position INTEGER NOT NULL,
    role TEXT DEFAULT 'author' CHECK(role IN ('author','editor','translator','advisor')),
    PRIMARY KEY(item_id, author_id, role)
);

CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE item_tags (
    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES tags(id),
    PRIMARY KEY(item_id, tag_id)
);

CREATE TABLE collections (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id INTEGER REFERENCES collections(id)
);

CREATE TABLE item_collections (
    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    collection_id INTEGER REFERENCES collections(id),
    PRIMARY KEY(item_id, collection_id)
);

CREATE TABLE item_relations (
    subject_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK(relation IN (
        'preprint_of', 'published_as',
        'erratum_for', 'corrected_by',
        'chapter_of', 'contains_chapter',
        'later_edition_of', 'earlier_edition_of',
        'translates', 'translated_from',
        'cites', 'cited_by',
        'related'
    )),
    object_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    confidence REAL DEFAULT 1.0,
    PRIMARY KEY(subject_id, relation, object_id)
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    page INTEGER,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(item_id, chunk_index)
);
CREATE INDEX idx_chunks_item ON chunks(item_id);

CREATE VIRTUAL TABLE items_fts USING fts5(
    title, abstract, notes,
    content='items', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

-- Keep FTS mirrors in sync with the content tables. Required for oracle check 8.
CREATE TRIGGER items_ai AFTER INSERT ON items BEGIN
    INSERT INTO items_fts(rowid, title, abstract, notes)
    VALUES (new.id, new.title, new.abstract, new.notes);
END;
CREATE TRIGGER items_ad AFTER DELETE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, title, abstract, notes)
    VALUES ('delete', old.id, old.title, old.abstract, old.notes);
END;
CREATE TRIGGER items_au AFTER UPDATE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, title, abstract, notes)
    VALUES ('delete', old.id, old.title, old.abstract, old.notes);
    INSERT INTO items_fts(rowid, title, abstract, notes)
    VALUES (new.id, new.title, new.abstract, new.notes);
END;

CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE VIRTUAL TABLE item_embeddings USING vec0(
    item_id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);
CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[1024]
);

CREATE TABLE merge_history (
    merged_id INTEGER PRIMARY KEY,
    target_id INTEGER REFERENCES items(id),
    reason TEXT,
    merged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE non_duplicate_pairs (
    a_id INTEGER NOT NULL,
    b_id INTEGER NOT NULL,
    asserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(a_id, b_id),
    CHECK(a_id < b_id)
);

-- Total rows ingested, for the conservation invariant (oracle check 1):
--   count(items) + count(merge_history) == count(ingest_log)
CREATE TABLE ingest_log (
    id INTEGER PRIMARY KEY,
    source_path TEXT,
    content_hash TEXT,
    result TEXT NOT NULL CHECK(result IN ('inserted','merged','linked','skipped','failed')),
    item_id INTEGER,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
