"""FTS5 keyword search over items_fts and chunks_fts."""

from __future__ import annotations

import re
import sqlite3

from grimoire.search.models import ItemHit, Snippet

# Pull out word-ish tokens (letters, digits, hyphens). Everything else becomes
# a separator — this strips FTS5 operators (AND/OR/NOT/NEAR, quotes, stars).
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-][A-Za-z0-9]+)*")


def _build_query(q: str) -> str:
    """Turn a free-form user query into a safe FTS5 MATCH expression.

    Each token is phrase-quoted so FTS5 operator syntax in the input can't
    reach the parser. The result is an implicit-AND conjunction of quoted
    terms — FTS5's default."""
    tokens = _TOKEN.findall(q)
    return " ".join(f'"{t}"' for t in tokens)


def search_items(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[ItemHit]:
    fts_query = _build_query(query)
    if not fts_query:
        return []
    rows = conn.execute(
        """SELECT rowid AS item_id, bm25(items_fts) AS score
           FROM items_fts
           WHERE items_fts MATCH ?
           ORDER BY score
           LIMIT ?""",
        (fts_query, limit),
    ).fetchall()
    return [ItemHit(item_id=int(r["item_id"]), score=float(r["score"])) for r in rows]


def search_chunks(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[Snippet]:
    fts_query = _build_query(query)
    if not fts_query:
        return []
    rows = conn.execute(
        """SELECT c.id AS chunk_id, c.item_id, c.page, c.text,
                  bm25(chunks_fts) AS score
           FROM chunks_fts
           JOIN chunks c ON c.id = chunks_fts.rowid
           WHERE chunks_fts MATCH ?
           ORDER BY score
           LIMIT ?""",
        (fts_query, limit),
    ).fetchall()
    return [
        Snippet(
            chunk_id=int(r["chunk_id"]),
            item_id=int(r["item_id"]),
            page=r["page"],
            text=r["text"],
            score=float(r["score"]),
        )
        for r in rows
    ]
