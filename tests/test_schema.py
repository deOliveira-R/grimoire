"""Oracle-style checks on the empty schema. Invariants that hold on every fresh DB
must remain true at all times — see plan §7."""

from __future__ import annotations

import sqlite3

import pytest

from grimoire import db as db_module


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,),
    ).fetchone()
    return row is not None


EXPECTED_TABLES = {
    "items",
    "authors",
    "item_authors",
    "tags",
    "item_tags",
    "collections",
    "item_collections",
    "item_relations",
    "chunks",
    "items_fts",
    "chunks_fts",
    "item_embeddings",
    "chunk_embeddings",
    "merge_history",
    "non_duplicate_pairs",
    "ingest_log",
}


def test_all_tables_exist(tmp_db: sqlite3.Connection) -> None:
    for name in EXPECTED_TABLES:
        assert _table_exists(tmp_db, name), f"missing table: {name}"


def test_apply_migrations_is_idempotent(tmp_db: sqlite3.Connection) -> None:
    new = db_module.apply_migrations(tmp_db)
    assert new == []


def test_item_embeddings_dim_768(tmp_db: sqlite3.Connection) -> None:
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('paper','t')")
    item_id = tmp_db.execute("SELECT id FROM items").fetchone()["id"]
    emb = b"\x00" * (4 * 768)
    tmp_db.execute("INSERT INTO item_embeddings(item_id, embedding) VALUES (?, ?)", (item_id, emb))
    with pytest.raises(sqlite3.OperationalError):
        bad = b"\x00" * (4 * 512)
        tmp_db.execute(
            "INSERT INTO item_embeddings(item_id, embedding) VALUES (?, ?)", (item_id + 1, bad)
        )


def test_chunk_embeddings_dim_1024(tmp_db: sqlite3.Connection) -> None:
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('paper','t')")
    item_id = tmp_db.execute("SELECT id FROM items").fetchone()["id"]
    tmp_db.execute("INSERT INTO chunks(item_id, chunk_index, text) VALUES (?, 0, 'x')", (item_id,))
    chunk_id = tmp_db.execute("SELECT id FROM chunks").fetchone()["id"]
    emb = b"\x00" * (4 * 1024)
    tmp_db.execute(
        "INSERT INTO chunk_embeddings(chunk_id, embedding) VALUES (?, ?)", (chunk_id, emb)
    )


def test_items_fts_trigger_sync(tmp_db: sqlite3.Connection) -> None:
    tmp_db.execute(
        "INSERT INTO items(item_type, title, abstract) VALUES ('paper','Alpha','beta gamma')"
    )
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('paper','Delta')")
    items = tmp_db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    fts = tmp_db.execute("SELECT COUNT(*) FROM items_fts").fetchone()[0]
    assert items == fts == 2

    hits = tmp_db.execute("SELECT rowid FROM items_fts WHERE items_fts MATCH 'gamma'").fetchall()
    assert len(hits) == 1


def test_chunks_fts_trigger_sync(tmp_db: sqlite3.Connection) -> None:
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('paper','t')")
    item_id = tmp_db.execute("SELECT id FROM items").fetchone()["id"]
    tmp_db.execute(
        "INSERT INTO chunks(item_id, chunk_index, text) VALUES (?, 0, 'lorem ipsum dolor')",
        (item_id,),
    )
    chunks = tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    fts = tmp_db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert chunks == fts == 1


def test_non_duplicate_pairs_canonical_ordering(tmp_db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        tmp_db.execute("INSERT INTO non_duplicate_pairs(a_id, b_id) VALUES (5, 3)")


def test_doi_unique_across_items(tmp_db: sqlite3.Connection) -> None:
    tmp_db.execute("INSERT INTO items(item_type, title, doi) VALUES ('paper','a','10.1/x')")
    with pytest.raises(sqlite3.IntegrityError):
        tmp_db.execute("INSERT INTO items(item_type, title, doi) VALUES ('paper','b','10.1/x')")


def test_content_hash_unique_when_set(tmp_db: sqlite3.Connection) -> None:
    tmp_db.execute("INSERT INTO items(item_type, title, content_hash) VALUES ('paper','a','abc')")
    with pytest.raises(sqlite3.IntegrityError):
        tmp_db.execute(
            "INSERT INTO items(item_type, title, content_hash) VALUES ('paper','b','abc')"
        )
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('paper','c')")
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('paper','d')")


def test_item_type_check_enforced(tmp_db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('novel','x')")


def test_fk_cascade_on_item_delete(tmp_db: sqlite3.Connection) -> None:
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('paper','t')")
    item_id = tmp_db.execute("SELECT id FROM items").fetchone()["id"]
    tmp_db.execute("INSERT INTO chunks(item_id, chunk_index, text) VALUES (?,0,'hi')", (item_id,))
    tmp_db.execute("DELETE FROM items WHERE id=?", (item_id,))
    assert tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert tmp_db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 0
