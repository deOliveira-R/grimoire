from __future__ import annotations

import sqlite3

from grimoire.search.keyword import search_chunks, search_items


def _insert_item(conn: sqlite3.Connection, title: str, abstract: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO items(item_type, title, abstract, content_hash) VALUES ('paper', ?, ?, NULL)",
        (title, abstract),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def test_items_fts_matches_title(tmp_db: sqlite3.Connection) -> None:
    a = _insert_item(tmp_db, "Reactor physics fundamentals")
    _insert_item(tmp_db, "Vegetable recipes")

    hits = search_items(tmp_db, "reactor", limit=10)
    assert [h.item_id for h in hits] == [a]
    assert hits[0].score < 0  # FTS5 bm25 returns negative, more-negative = better


def test_items_fts_matches_abstract(tmp_db: sqlite3.Connection) -> None:
    _insert_item(tmp_db, "Plain title", abstract="discussion of boron dilution transients")
    _insert_item(tmp_db, "Irrelevant", abstract="gardening tips")

    hits = search_items(tmp_db, "boron dilution", limit=10)
    assert len(hits) == 1


def test_items_fts_escapes_operators(tmp_db: sqlite3.Connection) -> None:
    """A raw FTS5 operator in user input must not crash the query."""
    _insert_item(tmp_db, "Study of OR-gate logic")

    # The word "OR" is an FTS5 operator; the wrapper must quote-escape.
    hits = search_items(tmp_db, "OR-gate", limit=10)
    assert len(hits) == 1


def test_chunks_fts(tmp_db: sqlite3.Connection) -> None:
    item_id = _insert_item(tmp_db, "Parent item")
    for i, text in enumerate(["boron dilution study", "heat transfer intro", "fuel rod geometry"]):
        tmp_db.execute(
            "INSERT INTO chunks(item_id, chunk_index, page, text) VALUES (?, ?, ?, ?)",
            (item_id, i, i + 1, text),
        )

    hits = search_chunks(tmp_db, "boron", limit=10)
    assert len(hits) == 1
    assert hits[0].page == 1
    assert "boron" in hits[0].text.lower()


def test_no_hits_returns_empty(tmp_db: sqlite3.Connection) -> None:
    _insert_item(tmp_db, "A paper")
    assert search_items(tmp_db, "xyzzy nonexistent", limit=5) == []
    assert search_chunks(tmp_db, "xyzzy nonexistent", limit=5) == []


def test_empty_query_returns_empty(tmp_db: sqlite3.Connection) -> None:
    _insert_item(tmp_db, "A paper")
    assert search_items(tmp_db, "", limit=5) == []
    assert search_items(tmp_db, "   ", limit=5) == []
