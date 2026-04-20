"""Pure-implementation MCP tool functions.

Each function takes an open ``sqlite3.Connection`` and returns a pydantic
model (or list of models). The MCP server module wraps them with FastMCP
``@tool`` decorators; tests call them directly against a seeded DB."""

from __future__ import annotations

import sqlite3
from typing import Literal

from grimoire.embed.base import Embedder, l2_normalize
from grimoire.mcp.models import (
    Collection,
    ItemFull,
    ItemSummary,
    RelatedItem,
    Snippet,
)
from grimoire.search import keyword as keyword_search
from grimoire.search import search_items
from grimoire.search import semantic as semantic_search

SearchMode = Literal["hybrid", "keyword", "semantic"]


# ---------- search / retrieval -----------------------------------------------


def search(
    conn: sqlite3.Connection,
    query: str,
    mode: SearchMode = "hybrid",
    item_types: list[str] | None = None,
    limit: int = 20,
    *,
    item_embedder: Embedder | None = None,
    chunk_embedder: Embedder | None = None,
) -> list[ItemSummary]:
    hits = search_items(
        conn,
        query,
        mode=mode,
        limit=limit * (2 if item_types else 1),
        item_embedder=item_embedder,
        chunk_embedder=chunk_embedder,
    )

    allowed = set(item_types) if item_types else None
    out: list[ItemSummary] = []
    for h in hits:
        row = conn.execute(
            "SELECT item_type, doi, arxiv_id FROM items WHERE id=?", (h.item_id,)
        ).fetchone()
        if row is None:
            continue
        if allowed and row["item_type"] not in allowed:
            continue
        authors = _item_authors(conn, h.item_id)
        snippet = (
            Snippet(
                item_id=h.snippet.item_id,
                chunk_id=h.snippet.chunk_id,
                page=h.snippet.page,
                chunk_index=_chunk_index(conn, h.snippet.chunk_id),
                text=h.snippet.text,
                score=h.snippet.score,
            )
            if h.snippet is not None
            else None
        )
        out.append(
            ItemSummary(
                item_id=h.item_id,
                title=h.title,
                year=h.year,
                authors=authors,
                venue=None,
                doi=row["doi"],
                arxiv_id=row["arxiv_id"],
                item_type=row["item_type"],
                score=h.score,
                snippet=snippet,
            )
        )
        if len(out) >= limit:
            break
    return out


def get_item(conn: sqlite3.Connection, item_id: int) -> ItemFull | None:
    row = conn.execute(
        """SELECT id, item_type, title, abstract, publication_year,
                  doi, arxiv_id, isbn, venue, volume, issue, pages,
                  series, edition, language, metadata_source, metadata_confidence
           FROM items WHERE id=?""",
        (item_id,),
    ).fetchone()
    if row is None:
        return None
    return ItemFull(
        item_id=int(row["id"]),
        title=row["title"],
        year=row["publication_year"],
        authors=_item_authors(conn, item_id),
        venue=row["venue"],
        doi=row["doi"],
        arxiv_id=row["arxiv_id"],
        isbn=row["isbn"],
        item_type=row["item_type"],
        abstract=row["abstract"],
        volume=row["volume"],
        issue=row["issue"],
        pages=row["pages"],
        series=row["series"],
        edition=row["edition"],
        language=row["language"],
        tags=_item_tags(conn, item_id),
        collections=_item_collections(conn, item_id),
        metadata_source=row["metadata_source"],
        metadata_confidence=row["metadata_confidence"],
    )


def get_full_text(conn: sqlite3.Connection, item_id: int, page: int | None = None) -> str:
    """Return full body text reconstructed from the item's chunks. When `page`
    is given, return only chunks whose `page` column equals it."""
    if page is not None:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE item_id=? AND page=? ORDER BY chunk_index",
            (item_id, page),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE item_id=? ORDER BY chunk_index", (item_id,)
        ).fetchall()
    return "\n\n".join(r["text"] for r in rows)


def get_snippets(
    conn: sqlite3.Connection,
    query: str,
    item_id: int | None = None,
    k: int = 10,
    *,
    chunk_embedder: Embedder | None = None,
) -> list[Snippet]:
    """Return the top-k best-matching chunks for ``query``. When ``item_id``
    is given, only chunks from that item. Uses semantic search when the
    chunk embedder is provided; falls back to FTS5 keyword otherwise."""
    if not query.strip():
        return []

    results: list[Snippet] = []
    if chunk_embedder is not None:
        vec = l2_normalize(chunk_embedder.encode([query]))[0]
        sem_hits = semantic_search.search_chunks_by_embedding(conn, vec, limit=k * 3)
        for h in sem_hits:
            if item_id is not None and h.item_id != item_id:
                continue
            results.append(
                Snippet(
                    item_id=h.item_id,
                    chunk_id=h.chunk_id,
                    page=h.page,
                    chunk_index=_chunk_index(conn, h.chunk_id),
                    text=h.text,
                    score=h.score,
                )
            )
            if len(results) >= k:
                break
        if results:
            return results

    kw_hits = keyword_search.search_chunks(conn, query, limit=k * 3)
    for h in kw_hits:
        if item_id is not None and h.item_id != item_id:
            continue
        results.append(
            Snippet(
                item_id=h.item_id,
                chunk_id=h.chunk_id,
                page=h.page,
                chunk_index=_chunk_index(conn, h.chunk_id),
                text=h.text,
                score=h.score,
            )
        )
        if len(results) >= k:
            break
    return results


RelationKind = Literal["all", "preprint_chain", "structural", "semantic", "citations"]


def list_related(
    conn: sqlite3.Connection, item_id: int, kind: RelationKind = "all"
) -> list[RelatedItem]:
    filters: tuple[str, ...]
    if kind == "preprint_chain":
        filters = ("preprint_of", "published_as", "later_edition_of", "earlier_edition_of")
    elif kind == "structural":
        # Chapter↔book and volume↔set (plan §6 Phase 6 part_of / contains_part).
        filters = ("chapter_of", "contains_chapter", "part_of", "contains_part")
    elif kind == "semantic":
        filters = ("related",)
    elif kind == "citations":
        filters = ("cites", "cited_by")
    else:
        filters = ()

    if filters:
        placeholders = ",".join("?" * len(filters))
        rows = conn.execute(
            f"""SELECT r.relation, r.object_id, r.confidence,
                       i.item_type, i.title, i.publication_year, i.doi, i.arxiv_id
                FROM item_relations r
                JOIN items i ON i.id = r.object_id
                WHERE r.subject_id = ? AND r.relation IN ({placeholders})""",
            (item_id, *filters),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT r.relation, r.object_id, r.confidence,
                      i.item_type, i.title, i.publication_year, i.doi, i.arxiv_id
               FROM item_relations r
               JOIN items i ON i.id = r.object_id
               WHERE r.subject_id = ?""",
            (item_id,),
        ).fetchall()
    out = []
    for r in rows:
        out.append(
            RelatedItem(
                item_id=int(r["object_id"]),
                title=r["title"],
                year=r["publication_year"],
                authors=_item_authors(conn, r["object_id"]),
                doi=r["doi"],
                arxiv_id=r["arxiv_id"],
                item_type=r["item_type"],
                relation=r["relation"],
                confidence=r["confidence"] if r["confidence"] is not None else 1.0,
            )
        )
    return out


def list_tags(conn: sqlite3.Connection) -> list[str]:
    return [row["name"] for row in conn.execute("SELECT name FROM tags ORDER BY name")]


def list_collections(conn: sqlite3.Connection) -> list[Collection]:
    rows = conn.execute(
        """SELECT c.id, c.name, c.parent_id, COUNT(ic.item_id) AS item_count
           FROM collections c
           LEFT JOIN item_collections ic ON ic.collection_id = c.id
           GROUP BY c.id
           ORDER BY c.name"""
    ).fetchall()
    return [
        Collection(
            id=int(r["id"]),
            name=r["name"],
            parent_id=r["parent_id"],
            item_count=int(r["item_count"] or 0),
        )
        for r in rows
    ]


def get_document_structure(
    conn: sqlite3.Connection, item_id: int
) -> dict[str, object] | None:
    """Return the parsed structure of an item's GROBID TEI artifact.

    Shape:
        {
          "header":     {"title", "abstract", "doi", "year", "authors": [...]},
          "sections":   [{"level", "heading", "text"}, ...],
          "references": [{"title", "authors", "year", "doi", "venue", "raw"}, ...]
        }

    Returns ``None`` when no TEI artifact exists for the item (not yet
    processed — run ``grimoire artifacts build --kind grobid_tei``)."""
    from grimoire.extract import tei as tei_parser
    from grimoire.storage import artifacts

    data = artifacts.read(conn, item_id, "grobid_tei")
    if data is None:
        return None
    return tei_parser.parse_structure(data)


def find_by_tag(conn: sqlite3.Connection, tag: str, limit: int = 100) -> list[ItemSummary]:
    rows = conn.execute(
        """SELECT i.id, i.item_type, i.title, i.publication_year, i.doi, i.arxiv_id, i.venue
           FROM items i
           JOIN item_tags it ON it.item_id = i.id
           JOIN tags t ON t.id = it.tag_id
           WHERE t.name = ?
           ORDER BY i.added_at DESC
           LIMIT ?""",
        (tag, limit),
    ).fetchall()
    return [
        ItemSummary(
            item_id=int(r["id"]),
            title=r["title"],
            year=r["publication_year"],
            authors=_item_authors(conn, r["id"]),
            venue=r["venue"],
            doi=r["doi"],
            arxiv_id=r["arxiv_id"],
            item_type=r["item_type"],
        )
        for r in rows
    ]


# ---------- internal helpers -------------------------------------------------


def _item_authors(conn: sqlite3.Connection, item_id: int) -> list[str]:
    rows = conn.execute(
        """SELECT a.family_name FROM item_authors ia
           JOIN authors a ON a.id = ia.author_id
           WHERE ia.item_id = ? ORDER BY ia.position""",
        (item_id,),
    ).fetchall()
    return [r["family_name"] for r in rows if r["family_name"]]


def _item_tags(conn: sqlite3.Connection, item_id: int) -> list[str]:
    return [
        row["name"]
        for row in conn.execute(
            """SELECT t.name FROM item_tags it JOIN tags t ON t.id = it.tag_id
               WHERE it.item_id = ? ORDER BY t.name""",
            (item_id,),
        )
    ]


def _item_collections(conn: sqlite3.Connection, item_id: int) -> list[str]:
    return [
        row["name"]
        for row in conn.execute(
            """SELECT c.name FROM item_collections ic JOIN collections c ON c.id = ic.collection_id
               WHERE ic.item_id = ? ORDER BY c.name""",
            (item_id,),
        )
    ]


def _chunk_index(conn: sqlite3.Connection, chunk_id: int) -> int:
    row = conn.execute("SELECT chunk_index FROM chunks WHERE id=?", (chunk_id,)).fetchone()
    return int(row["chunk_index"]) if row else 0
