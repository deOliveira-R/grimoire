"""Shared DB query helpers for the web layer (OPDS + Phase 5b HTML UI).

These return lightweight dataclasses tuned for listing views. MCP's pydantic
models stay untouched — feed rendering needs fields (content_hash, added_at,
language, file_path) that JSON clients don't care about."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class FeedItem:
    item_id: int
    item_type: str
    title: str
    abstract: str | None
    year: int | None
    venue: str | None
    volume: str | None
    issue: str | None
    pages: str | None
    doi: str | None
    arxiv_id: str | None
    isbn: str | None
    language: str | None
    content_hash: str | None
    file_path: str | None
    added_at: str
    authors: list[str] = field(default_factory=list)


@dataclass
class TagCount:
    name: str
    item_count: int


@dataclass
class AuthorRow:
    author_id: int
    family_name: str
    given_name: str | None
    item_count: int


@dataclass
class CollectionRow:
    collection_id: int
    name: str
    parent_id: int | None
    item_count: int


@dataclass
class CollectionTreeNode:
    collection: CollectionRow
    children: list["CollectionTreeNode"] = field(default_factory=list)
    descendants_count: int = 0


_FEED_COLS = (
    "i.id, i.item_type, i.title, i.abstract, i.publication_year, "
    "i.venue, i.volume, i.issue, i.pages, "
    "i.doi, i.arxiv_id, i.isbn, i.language, i.content_hash, i.file_path, i.added_at"
)


def _to_item(row: sqlite3.Row, authors: list[str]) -> FeedItem:
    return FeedItem(
        item_id=int(row["id"]),
        item_type=row["item_type"],
        title=row["title"],
        abstract=row["abstract"],
        year=row["publication_year"],
        venue=row["venue"],
        volume=row["volume"],
        issue=row["issue"],
        pages=row["pages"],
        doi=row["doi"],
        arxiv_id=row["arxiv_id"],
        isbn=row["isbn"],
        language=row["language"],
        content_hash=row["content_hash"],
        file_path=row["file_path"],
        added_at=row["added_at"],
        authors=authors,
    )


def _authors_for(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, list[str]]:
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""SELECT ia.item_id, a.family_name, a.given_name
            FROM item_authors ia
            JOIN authors a ON a.id = ia.author_id
            WHERE ia.item_id IN ({placeholders}) AND ia.role = 'author'
            ORDER BY ia.item_id, ia.position""",
        tuple(ids),
    ).fetchall()
    out: dict[int, list[str]] = {}
    for r in rows:
        given = (r["given_name"] or "").strip()
        family = r["family_name"]
        name = f"{given} {family}".strip() if given else family
        out.setdefault(int(r["item_id"]), []).append(name)
    return out


def _hydrate(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> list[FeedItem]:
    ids = [int(r["id"]) for r in rows]
    authors_by_id = _authors_for(conn, ids)
    return [_to_item(r, authors_by_id.get(int(r["id"]), [])) for r in rows]


# ---- listings ---------------------------------------------------------------


def list_recent(conn: sqlite3.Connection, *, offset: int, limit: int) -> list[FeedItem]:
    rows = conn.execute(
        f"SELECT {_FEED_COLS} FROM items i ORDER BY i.added_at DESC, i.id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return _hydrate(conn, rows)


def count_all(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"])


def list_by_type(
    conn: sqlite3.Connection, item_type: str, *, offset: int, limit: int
) -> list[FeedItem]:
    rows = conn.execute(
        f"SELECT {_FEED_COLS} FROM items i WHERE i.item_type = ? "
        "ORDER BY i.added_at DESC, i.id DESC LIMIT ? OFFSET ?",
        (item_type, limit, offset),
    ).fetchall()
    return _hydrate(conn, rows)


def count_by_type(conn: sqlite3.Connection, item_type: str) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM items WHERE item_type = ?", (item_type,)
        ).fetchone()["n"]
    )


def list_in_collection(
    conn: sqlite3.Connection, collection_id: int, *, offset: int, limit: int
) -> list[FeedItem]:
    rows = conn.execute(
        f"SELECT {_FEED_COLS} FROM items i "
        "JOIN item_collections ic ON ic.item_id = i.id "
        "WHERE ic.collection_id = ? "
        "ORDER BY i.added_at DESC, i.id DESC LIMIT ? OFFSET ?",
        (collection_id, limit, offset),
    ).fetchall()
    return _hydrate(conn, rows)


def count_in_collection(conn: sqlite3.Connection, collection_id: int) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM item_collections WHERE collection_id = ?",
            (collection_id,),
        ).fetchone()["n"]
    )


def list_with_tag(
    conn: sqlite3.Connection, tag: str, *, offset: int, limit: int
) -> list[FeedItem]:
    rows = conn.execute(
        f"SELECT {_FEED_COLS} FROM items i "
        "JOIN item_tags it ON it.item_id = i.id "
        "JOIN tags t ON t.id = it.tag_id "
        "WHERE t.name = ? "
        "ORDER BY i.added_at DESC, i.id DESC LIMIT ? OFFSET ?",
        (tag, limit, offset),
    ).fetchall()
    return _hydrate(conn, rows)


def count_with_tag(conn: sqlite3.Connection, tag: str) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM item_tags it "
            "JOIN tags t ON t.id = it.tag_id WHERE t.name = ?",
            (tag,),
        ).fetchone()["n"]
    )


def list_by_venue(
    conn: sqlite3.Connection, venue: str, *, offset: int, limit: int
) -> list[FeedItem]:
    """Items in a journal/venue, ordered for a typical bibliographic browse:
    newest issues first, then descending volume/issue within a year."""
    rows = conn.execute(
        f"SELECT {_FEED_COLS} FROM items i WHERE i.venue = ? "
        "ORDER BY i.publication_year DESC, "
        "         CAST(i.volume AS INTEGER) DESC, "
        "         CAST(i.issue AS INTEGER) DESC, "
        "         CAST(SUBSTR(i.pages, 1, INSTR(i.pages || '-', '-') - 1) AS INTEGER) ASC, "
        "         i.id DESC "
        "LIMIT ? OFFSET ?",
        (venue, limit, offset),
    ).fetchall()
    return _hydrate(conn, rows)


def count_by_venue(conn: sqlite3.Connection, venue: str) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM items WHERE venue = ?", (venue,)
        ).fetchone()["n"]
    )


def list_by_year(
    conn: sqlite3.Connection, year: int, *, offset: int, limit: int
) -> list[FeedItem]:
    rows = conn.execute(
        f"SELECT {_FEED_COLS} FROM items i WHERE i.publication_year = ? "
        "ORDER BY i.venue, "
        "         CAST(i.volume AS INTEGER) DESC, "
        "         CAST(i.issue AS INTEGER) DESC, "
        "         i.id DESC "
        "LIMIT ? OFFSET ?",
        (year, limit, offset),
    ).fetchall()
    return _hydrate(conn, rows)


def count_by_year(conn: sqlite3.Connection, year: int) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM items WHERE publication_year = ?", (year,)
        ).fetchone()["n"]
    )


def list_by_author(
    conn: sqlite3.Connection, author_id: int, *, offset: int, limit: int
) -> list[FeedItem]:
    rows = conn.execute(
        f"SELECT {_FEED_COLS} FROM items i "
        "JOIN item_authors ia ON ia.item_id = i.id "
        "WHERE ia.author_id = ? "
        "ORDER BY i.added_at DESC, i.id DESC LIMIT ? OFFSET ?",
        (author_id, limit, offset),
    ).fetchall()
    return _hydrate(conn, rows)


def count_by_author(conn: sqlite3.Connection, author_id: int) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM item_authors WHERE author_id = ?", (author_id,)
        ).fetchone()["n"]
    )


# ---- stackable filter (web UI) ---------------------------------------------


_SORT_ORDERS: dict[str, str] = {
    # Stable tie-breaker on ``i.id DESC`` so repeated scrolls are consistent.
    "added": "i.added_at DESC, i.id DESC",
    "year": "i.publication_year DESC NULLS LAST, i.id DESC",
    "year_asc": "i.publication_year ASC NULLS LAST, i.id ASC",
    "title": "LOWER(i.title) ASC, i.id ASC",
    "author": (
        "(SELECT LOWER(a.family_name) FROM item_authors ia "
        "JOIN authors a ON a.id = ia.author_id "
        "WHERE ia.item_id = i.id AND ia.role = 'author' "
        "ORDER BY ia.position LIMIT 1) ASC NULLS LAST, i.id ASC"
    ),
}
DEFAULT_SORT = "added"


def _filter_where(
    *,
    item_type: str | None,
    venue: str | None,
    year: int | None,
    tag: str | None,
    collection_id: int | None,
) -> tuple[str, list[object], str]:
    """Build a WHERE fragment + params + any JOIN needed for the given filters.

    Returns ``(where_sql, params, extra_joins)``. ``where_sql`` never starts
    with ``WHERE`` — caller prepends it if non-empty."""
    clauses: list[str] = []
    params: list[object] = []
    joins: list[str] = []
    if item_type:
        clauses.append("i.item_type = ?")
        params.append(item_type)
    if venue:
        clauses.append("i.venue = ?")
        params.append(venue)
    if year is not None:
        clauses.append("i.publication_year = ?")
        params.append(year)
    if tag:
        joins.append(
            " JOIN item_tags it ON it.item_id = i.id "
            " JOIN tags t ON t.id = it.tag_id "
        )
        clauses.append("t.name = ?")
        params.append(tag)
    if collection_id is not None:
        joins.append(
            " JOIN item_collections ic ON ic.item_id = i.id "
        )
        clauses.append("ic.collection_id = ?")
        params.append(collection_id)
    where_sql = " AND ".join(clauses)
    return where_sql, params, " ".join(joins)


def list_filtered(
    conn: sqlite3.Connection,
    *,
    item_type: str | None = None,
    venue: str | None = None,
    year: int | None = None,
    tag: str | None = None,
    collection_id: int | None = None,
    sort: str = DEFAULT_SORT,
    offset: int,
    limit: int,
) -> list[FeedItem]:
    where, params, joins = _filter_where(
        item_type=item_type, venue=venue, year=year, tag=tag, collection_id=collection_id
    )
    order = _SORT_ORDERS.get(sort, _SORT_ORDERS[DEFAULT_SORT])
    sql = f"SELECT {_FEED_COLS} FROM items i {joins}"
    if where:
        sql += f" WHERE {where}"
    sql += f" ORDER BY {order} LIMIT ? OFFSET ?"
    rows = conn.execute(sql, (*params, limit, offset)).fetchall()
    return _hydrate(conn, rows)


def count_filtered(
    conn: sqlite3.Connection,
    *,
    item_type: str | None = None,
    venue: str | None = None,
    year: int | None = None,
    tag: str | None = None,
    collection_id: int | None = None,
) -> int:
    where, params, joins = _filter_where(
        item_type=item_type, venue=venue, year=year, tag=tag, collection_id=collection_id
    )
    sql = f"SELECT COUNT(*) AS n FROM items i {joins}"
    if where:
        sql += f" WHERE {where}"
    return int(conn.execute(sql, tuple(params)).fetchone()["n"])


# ---- facet listings ---------------------------------------------------------


def list_collections(conn: sqlite3.Connection) -> list[CollectionRow]:
    rows = conn.execute(
        """SELECT c.id, c.name, c.parent_id, COUNT(ic.item_id) AS item_count
           FROM collections c
           LEFT JOIN item_collections ic ON ic.collection_id = c.id
           GROUP BY c.id
           ORDER BY c.name"""
    ).fetchall()
    return [
        CollectionRow(
            collection_id=int(r["id"]),
            name=r["name"],
            parent_id=r["parent_id"],
            item_count=int(r["item_count"] or 0),
        )
        for r in rows
    ]


def list_collections_tree(conn: sqlite3.Connection) -> list[CollectionTreeNode]:
    """Build the collection forest from the flat list, rolling up descendant
    counts so a parent reports ``item_count`` of itself plus all descendants.

    Cycles in ``parent_id`` (shouldn't happen, but the schema doesn't forbid
    them) are broken by skipping any node that would reattach under itself."""
    flat = list_collections(conn)
    by_id = {c.collection_id: CollectionTreeNode(c) for c in flat}
    roots: list[CollectionTreeNode] = []
    for c in flat:
        node = by_id[c.collection_id]
        parent = by_id.get(c.parent_id) if c.parent_id is not None else None
        if parent is not None and parent is not node:
            parent.children.append(node)
        else:
            roots.append(node)

    def _roll_up(node: CollectionTreeNode) -> int:
        node.descendants_count = node.collection.item_count + sum(
            _roll_up(ch) for ch in node.children
        )
        return node.descendants_count

    def _sort(node: CollectionTreeNode) -> None:
        node.children.sort(key=lambda ch: ch.collection.name.lower())
        for ch in node.children:
            _sort(ch)

    for r in roots:
        _roll_up(r)
    roots.sort(key=lambda n: n.collection.name.lower())
    for r in roots:
        _sort(r)
    return roots


def get_collection(conn: sqlite3.Connection, collection_id: int) -> CollectionRow | None:
    row = conn.execute(
        "SELECT id, name, parent_id FROM collections WHERE id = ?", (collection_id,)
    ).fetchone()
    if row is None:
        return None
    return CollectionRow(
        collection_id=int(row["id"]),
        name=row["name"],
        parent_id=row["parent_id"],
        item_count=count_in_collection(conn, collection_id),
    )


@dataclass
class VenueCount:
    venue: str
    item_count: int


@dataclass
class YearCount:
    year: int
    item_count: int


def list_venues_with_counts(
    conn: sqlite3.Connection, *, limit: int = 500
) -> list[VenueCount]:
    rows = conn.execute(
        """SELECT venue, COUNT(*) AS n FROM items
           WHERE venue IS NOT NULL AND TRIM(venue) <> ''
           GROUP BY venue ORDER BY n DESC, venue LIMIT ?""",
        (limit,),
    ).fetchall()
    return [VenueCount(venue=r["venue"], item_count=int(r["n"])) for r in rows]


def list_years_with_counts(conn: sqlite3.Connection) -> list[YearCount]:
    rows = conn.execute(
        """SELECT publication_year AS y, COUNT(*) AS n FROM items
           WHERE publication_year IS NOT NULL
           GROUP BY y ORDER BY y DESC"""
    ).fetchall()
    return [YearCount(year=int(r["y"]), item_count=int(r["n"])) for r in rows]


def list_tags_with_counts(conn: sqlite3.Connection) -> list[TagCount]:
    rows = conn.execute(
        """SELECT t.name, COUNT(it.item_id) AS item_count
           FROM tags t
           LEFT JOIN item_tags it ON it.tag_id = t.id
           GROUP BY t.id
           ORDER BY t.name"""
    ).fetchall()
    return [TagCount(name=r["name"], item_count=int(r["item_count"] or 0)) for r in rows]


def list_authors(conn: sqlite3.Connection, *, limit: int = 1000) -> list[AuthorRow]:
    rows = conn.execute(
        """SELECT a.id, a.family_name, a.given_name, COUNT(ia.item_id) AS item_count
           FROM authors a
           JOIN item_authors ia ON ia.author_id = a.id AND ia.role = 'author'
           GROUP BY a.id
           HAVING item_count > 0
           ORDER BY a.family_name, a.given_name
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        AuthorRow(
            author_id=int(r["id"]),
            family_name=r["family_name"],
            given_name=r["given_name"],
            item_count=int(r["item_count"]),
        )
        for r in rows
    ]


def get_author(conn: sqlite3.Connection, author_id: int) -> AuthorRow | None:
    row = conn.execute(
        """SELECT a.id, a.family_name, a.given_name,
                  (SELECT COUNT(*) FROM item_authors ia WHERE ia.author_id = a.id) AS item_count
           FROM authors a WHERE a.id = ?""",
        (author_id,),
    ).fetchone()
    if row is None:
        return None
    return AuthorRow(
        author_id=int(row["id"]),
        family_name=row["family_name"],
        given_name=row["given_name"],
        item_count=int(row["item_count"] or 0),
    )


# ---- full item fetch (for detail page / search hydration) -------------------


def get_feed_item(conn: sqlite3.Connection, item_id: int) -> FeedItem | None:
    row = conn.execute(
        f"SELECT {_FEED_COLS} FROM items i WHERE i.id = ?", (item_id,)
    ).fetchone()
    if row is None:
        return None
    authors_by_id = _authors_for(conn, [item_id])
    return _to_item(row, authors_by_id.get(item_id, []))


def hydrate_by_ids(conn: sqlite3.Connection, ids: Sequence[int]) -> list[FeedItem]:
    """Fetch items in the order given by ``ids``. Unknown ids are dropped."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT {_FEED_COLS} FROM items i WHERE i.id IN ({placeholders})", tuple(ids)
    ).fetchall()
    by_id = {int(r["id"]): r for r in rows}
    authors_by_id = _authors_for(conn, [i for i in ids if i in by_id])
    out: list[FeedItem] = []
    for i in ids:
        r = by_id.get(int(i))
        if r is not None:
            out.append(_to_item(r, authors_by_id.get(int(i), [])))
    return out
