"""Minimal HTML web UI — item browse + item detail. Jinja2 templates, no JS.

Scope (plan §6 Phase 5 — trimmed):
  GET /             — item list + facets (type, tag) + keyword search
  GET /items/{id}   — detail page with related items and download link

The ``/upload`` and ``/submit-url`` surfaces in the original plan are deferred
to v1.1; CLI ``grimoire ingest`` covers the ingestion path."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from grimoire.db import apply_migrations, connect
from grimoire.mcp.citation import to_bibtex
from grimoire.mcp.tools import list_related as list_related_items
from grimoire.search import keyword as keyword_search
from grimoire.web import queries

router = APIRouter(tags=["web"])

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _db() -> Iterator[sqlite3.Connection]:
    conn = connect()
    apply_migrations(conn)
    try:
        yield conn
    finally:
        conn.close()


ITEM_TYPES = (
    "paper",
    "book",
    "chapter",
    "report",
    "thesis",
    "preprint",
    "standard",
    "patent",
    "other",
)


_SORT_LABELS: dict[str, str] = {
    "added": "Recently added",
    "year": "Year (newest)",
    "year_asc": "Year (oldest)",
    "title": "Title (A–Z)",
}


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    q: str | None = Query(None),
    type: str | None = Query(None, alias="type"),
    tag: str | None = Query(None),
    venue: str | None = Query(None),
    year: int | None = Query(None),
    collection: int | None = Query(None),
    sort: str = Query(queries.DEFAULT_SORT),
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(_db),
) -> HTMLResponse:
    # Search short-circuits the other filters: FTS results drive the list
    # and only the search query is shown in facets.
    if q:
        hits = keyword_search.search_items(conn, q, limit=offset + limit)
        total = len(hits)
        items = queries.hydrate_by_ids(
            conn, [h.item_id for h in hits[offset : offset + limit]]
        )
    else:
        active_type = type if type in ITEM_TYPES else None
        items = queries.list_filtered(
            conn,
            item_type=active_type,
            venue=venue,
            year=year,
            tag=tag,
            collection_id=collection,
            sort=sort,
            offset=offset,
            limit=limit,
        )
        total = queries.count_filtered(
            conn,
            item_type=active_type,
            venue=venue,
            year=year,
            tag=tag,
            collection_id=collection,
        )

    type_counts = [(t, queries.count_by_type(conn, t)) for t in ITEM_TYPES]
    type_counts = [(t, n) for t, n in type_counts if n > 0]
    tag_counts = [t for t in queries.list_tags_with_counts(conn) if t.item_count > 0]
    venue_counts = queries.list_venues_with_counts(conn, limit=50)
    year_counts = queries.list_years_with_counts(conn)
    collections = [c for c in queries.list_collections(conn) if c.item_count > 0]

    # Look up the active collection's name for breadcrumb display.
    active_collection_name: str | None = None
    if collection is not None:
        col = queries.get_collection(conn, collection)
        active_collection_name = col.name if col else None

    active_filters: dict[str, str] = {}
    if q:
        active_filters["q"] = q
    if type and type in ITEM_TYPES:
        active_filters["type"] = type
    if tag:
        active_filters["tag"] = tag
    if venue:
        active_filters["venue"] = venue
    if year is not None:
        active_filters["year"] = str(year)
    if collection is not None and active_collection_name:
        active_filters["collection"] = str(collection)

    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit,
            "q": q or "",
            "active_type": type,
            "active_tag": tag,
            "active_venue": venue,
            "active_year": year,
            "active_collection": collection,
            "active_collection_name": active_collection_name,
            "active_filters": active_filters,
            "type_counts": type_counts,
            "tag_counts": tag_counts,
            "venue_counts": venue_counts,
            "year_counts": year_counts,
            "collections": collections,
            "sort": sort if sort in _SORT_LABELS else queries.DEFAULT_SORT,
            "sort_labels": _SORT_LABELS,
        },
    )


@router.get("/items/{item_id}", response_class=HTMLResponse)
def item_detail(
    item_id: int,
    request: Request,
    conn: sqlite3.Connection = Depends(_db),
) -> HTMLResponse:
    item = queries.get_feed_item(conn, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    tags = [
        row["name"]
        for row in conn.execute(
            "SELECT t.name FROM item_tags it JOIN tags t ON t.id = it.tag_id "
            "WHERE it.item_id = ? ORDER BY t.name",
            (item_id,),
        )
    ]
    collections = [
        {"id": int(row["id"]), "name": row["name"]}
        for row in conn.execute(
            "SELECT c.id, c.name FROM item_collections ic JOIN collections c "
            "ON c.id = ic.collection_id WHERE ic.item_id = ? ORDER BY c.name",
            (item_id,),
        )
    ]
    related = list_related_items(conn, item_id)
    # Split related into chapter-list vs everything else so the template
    # can render chapters inline as a table of contents.
    chapters = [r for r in related if r.relation == "contains_chapter"]
    other_related = [r for r in related if r.relation != "contains_chapter"]
    bibtex = to_bibtex(conn, item_id)
    return templates.TemplateResponse(
        request=request,
        name="item.html",
        context={
            "item": item,
            "tags": tags,
            "collections": collections,
            "related": other_related,
            "chapters": chapters,
            "bibtex": bibtex,
        },
    )


@router.get("/items/{item_id}/bibtex", response_class=Response)
def item_bibtex(
    item_id: int,
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    """Serve the BibTeX as plain text so ``curl -O`` and "Save As…" in the
    browser produce a usable ``.bib`` file."""
    bib = to_bibtex(conn, item_id)
    if bib is None:
        raise HTTPException(status_code=404, detail="item not found")
    return Response(
        content=bib,
        media_type="application/x-bibtex",
        headers={"Content-Disposition": f'inline; filename="grimoire-{item_id}.bib"'},
    )
