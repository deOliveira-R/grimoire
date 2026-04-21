"""Invariants from plan §7 that aren't already covered by test_schema.py.

- Invariant 7: `item_type='paper' AND doi IS NOT NULL` ⇒ `venue IS NOT NULL`
  (aspirational; fails over a threshold to detect regression without flagging
  legitimately-missing Crossref venues).
- Invariant 10: no orphan chunks — FK cascade enforces it at delete time,
  but the query itself must still detect force-inserted orphans.
"""

from __future__ import annotations

import sqlite3


_VENUE_BACKFILL_VIOLATION_THRESHOLD = 0.05


def test_invariant_7_venue_backfill_rate(tmp_db: sqlite3.Connection) -> None:
    # Mix of valid (paper+DOI+venue) and violating (paper+DOI, no venue) rows.
    tmp_db.executemany(
        "INSERT INTO items(item_type, title, doi, venue) VALUES (?,?,?,?)",
        [
            ("paper", "A", "10.1/a", "Nature"),
            ("paper", "B", "10.1/b", "Science"),
            ("paper", "C", "10.1/c", "PRL"),
            ("paper", "D", "10.1/d", "JACS"),
            # One violation: paper with DOI but no venue.
            ("paper", "E", "10.1/e", None),
            # Non-paper and paper-without-DOI rows are excluded from the ratio.
            ("book", "F", None, None),
            ("paper", "G", None, None),
        ],
    )
    violations = tmp_db.execute(
        "SELECT COUNT(*) FROM items "
        "WHERE item_type='paper' AND doi IS NOT NULL AND venue IS NULL"
    ).fetchone()[0]
    total = tmp_db.execute(
        "SELECT COUNT(*) FROM items WHERE item_type='paper' AND doi IS NOT NULL"
    ).fetchone()[0]
    assert total == 5
    assert violations == 1
    # 1/5 = 20 % > threshold → documents what a regression would look like.
    assert violations / total > _VENUE_BACKFILL_VIOLATION_THRESHOLD


def test_invariant_7_clean_library_passes(tmp_db: sqlite3.Connection) -> None:
    """Real library shape: every paper+DOI has a venue."""
    tmp_db.executemany(
        "INSERT INTO items(item_type, title, doi, venue) VALUES (?,?,?,?)",
        [
            ("paper", "A", "10.1/a", "Nature"),
            ("paper", "B", "10.1/b", "Science"),
        ],
    )
    violations = tmp_db.execute(
        "SELECT COUNT(*) FROM items "
        "WHERE item_type='paper' AND doi IS NOT NULL AND venue IS NULL"
    ).fetchone()[0]
    total = tmp_db.execute(
        "SELECT COUNT(*) FROM items WHERE item_type='paper' AND doi IS NOT NULL"
    ).fetchone()[0]
    assert total == 0 or violations / total <= _VENUE_BACKFILL_VIOLATION_THRESHOLD


def test_invariant_10_no_orphan_chunks_clean_db(tmp_db: sqlite3.Connection) -> None:
    """FK cascade invariant: deleting an item wipes its chunks, so a clean DB
    has zero orphans."""
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('paper','t')")
    item_id = tmp_db.execute("SELECT id FROM items").fetchone()["id"]
    tmp_db.execute(
        "INSERT INTO chunks(item_id, chunk_index, text) VALUES (?,0,'a')", (item_id,)
    )
    tmp_db.execute(
        "INSERT INTO chunks(item_id, chunk_index, text) VALUES (?,1,'b')", (item_id,)
    )
    tmp_db.execute("DELETE FROM items WHERE id=?", (item_id,))

    orphans = tmp_db.execute(
        "SELECT COUNT(*) FROM chunks c "
        "LEFT JOIN items i ON i.id = c.item_id "
        "WHERE i.id IS NULL"
    ).fetchone()[0]
    assert orphans == 0


def test_invariant_10_orphan_query_detects_forced_orphan(
    tmp_db: sqlite3.Connection,
) -> None:
    """Belt-and-suspenders: if FK enforcement is off, orphans are detectable."""
    tmp_db.execute("PRAGMA foreign_keys=OFF")
    tmp_db.execute(
        "INSERT INTO chunks(item_id, chunk_index, text) VALUES (9999,0,'x')"
    )
    tmp_db.execute("PRAGMA foreign_keys=ON")

    orphans = tmp_db.execute(
        "SELECT COUNT(*) FROM chunks c "
        "LEFT JOIN items i ON i.id = c.item_id "
        "WHERE i.id IS NULL"
    ).fetchone()[0]
    assert orphans == 1
