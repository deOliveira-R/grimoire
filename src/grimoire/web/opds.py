"""OPDS 1.2 Atom catalog.

Root is a navigation feed. Child feeds are either navigation (collections,
tags, authors) or acquisition (item lists with download links to
``/files/{content_hash}``).

Spec: https://specs.opds.io/opds-1.2 — we cover the subset KOReader, Marvin,
and MapleRead actually ship against: navigation feeds, acquisition feeds,
OpenSearch, and ``next``/``prev`` pagination links."""

from __future__ import annotations

import mimetypes
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from xml.etree import ElementTree as ET

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from grimoire.db import apply_migrations, connect
from grimoire.search import keyword as keyword_search
from grimoire.web import queries
from grimoire.web.queries import FeedItem

router = APIRouter(prefix="/opds", tags=["opds"])

# OPDS-specific MIME types. Clients match on these to distinguish catalog kinds.
NAV_TYPE = "application/atom+xml;profile=opds-catalog;kind=navigation"
ACQ_TYPE = "application/atom+xml;profile=opds-catalog;kind=acquisition"
OPENSEARCH_TYPE = "application/opensearchdescription+xml"

ATOM_NS = "http://www.w3.org/2005/Atom"
DC_NS = "http://purl.org/dc/terms/"
OPDS_NS = "http://opds-spec.org/2010/catalog"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"

# Register namespace prefixes once so ET emits tidy xmlns declarations on
# the root element instead of ns0/ns1/ns2 placeholders.
ET.register_namespace("", ATOM_NS)
ET.register_namespace("dc", DC_NS)
ET.register_namespace("opds", OPDS_NS)
ET.register_namespace("opensearch", OPENSEARCH_NS)

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

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200


def _db() -> Iterator[sqlite3.Connection]:
    conn = connect()
    apply_migrations(conn)
    try:
        yield conn
    finally:
        conn.close()


def _iso(ts: str | None = None) -> str:
    if ts is None:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    # SQLite CURRENT_TIMESTAMP returns 'YYYY-MM-DD HH:MM:SS' (no timezone).
    # Atom requires an offset; treat stored timestamps as UTC.
    return ts.replace(" ", "T") + ("Z" if not ts.endswith("Z") else "")


def _mime_for(item: FeedItem) -> str:
    if item.file_path:
        guessed, _ = mimetypes.guess_type(item.file_path)
        if guessed:
            return guessed
    return "application/octet-stream"


def _identifier(item: FeedItem) -> str:
    if item.doi:
        return f"urn:doi:{item.doi}"
    if item.arxiv_id:
        return f"urn:arxiv:{item.arxiv_id}"
    if item.isbn:
        return f"urn:isbn:{item.isbn}"
    if item.content_hash:
        return f"urn:sha256:{item.content_hash}"
    return f"urn:grimoire:item:{item.item_id}"


def _qname(prefix_ns: str, local: str) -> str:
    return f"{{{prefix_ns}}}{local}"


def _sub(parent: ET.Element, tag: str, text: str | None = None) -> ET.Element:
    elem = ET.SubElement(parent, tag)
    if text is not None:
        elem.text = text
    return elem


def _link(
    parent: ET.Element,
    *,
    rel: str,
    href: str,
    type: str | None = None,
    title: str | None = None,
) -> ET.Element:
    link = ET.SubElement(parent, _qname(ATOM_NS, "link"))
    link.set("rel", rel)
    link.set("href", href)
    if type is not None:
        link.set("type", type)
    if title is not None:
        link.set("title", title)
    return link


def _feed_skeleton(
    *,
    title: str,
    feed_id: str,
    self_path: str,
    updated: str | None = None,
) -> ET.Element:
    feed = ET.Element(_qname(ATOM_NS, "feed"))
    _sub(feed, _qname(ATOM_NS, "id"), feed_id)
    _sub(feed, _qname(ATOM_NS, "title"), title)
    _sub(feed, _qname(ATOM_NS, "updated"), updated or _iso())
    author = _sub(feed, _qname(ATOM_NS, "author"))
    _sub(author, _qname(ATOM_NS, "name"), "Grimoire")
    _link(feed, rel="self", href=self_path, type=NAV_TYPE)
    _link(feed, rel="start", href="/opds", type=NAV_TYPE)
    _link(
        feed,
        rel="search",
        href="/opds/opensearch.xml",
        type=OPENSEARCH_TYPE,
    )
    return feed


def _nav_entry(
    feed: ET.Element,
    *,
    title: str,
    href: str,
    summary: str,
    kind: str,
) -> None:
    entry = ET.SubElement(feed, _qname(ATOM_NS, "entry"))
    _sub(entry, _qname(ATOM_NS, "id"), f"urn:grimoire:nav:{href}")
    _sub(entry, _qname(ATOM_NS, "title"), title)
    _sub(entry, _qname(ATOM_NS, "updated"), _iso())
    _sub(entry, _qname(ATOM_NS, "content"), summary).set("type", "text")
    _link(entry, rel="subsection", href=href, type=kind, title=title)


def _acquisition_entry(feed: ET.Element, item: FeedItem) -> None:
    entry = ET.SubElement(feed, _qname(ATOM_NS, "entry"))
    _sub(entry, _qname(ATOM_NS, "id"), _identifier(item))
    _sub(entry, _qname(ATOM_NS, "title"), item.title)
    _sub(entry, _qname(ATOM_NS, "updated"), _iso(item.added_at))
    if item.abstract:
        summary = _sub(entry, _qname(ATOM_NS, "summary"), item.abstract)
        summary.set("type", "text")
    for name in item.authors:
        author = _sub(entry, _qname(ATOM_NS, "author"))
        _sub(author, _qname(ATOM_NS, "name"), name)
    if item.year is not None:
        _sub(entry, _qname(DC_NS, "issued"), str(item.year))
    if item.language:
        _sub(entry, _qname(DC_NS, "language"), item.language)
    if item.venue:
        _sub(entry, _qname(DC_NS, "source"), item.venue)
    _sub(entry, _qname(DC_NS, "identifier"), _identifier(item))
    category = ET.SubElement(entry, _qname(ATOM_NS, "category"))
    category.set("term", item.item_type)
    category.set("label", item.item_type.capitalize())
    if item.content_hash:
        _link(
            entry,
            rel="http://opds-spec.org/acquisition",
            href=f"/files/{item.content_hash}",
            type=_mime_for(item),
            title="Download",
        )
    _link(
        entry,
        rel="alternate",
        href=f"/items/{item.item_id}",
        type="text/html",
        title="Details",
    )


def _paginate(
    feed: ET.Element,
    *,
    base_path: str,
    offset: int,
    limit: int,
    total: int,
    query_prefix: str = "",
) -> None:
    """Attach OPDS pagination links (first/prev/next/last)."""

    def href(new_offset: int) -> str:
        sep = "&" if query_prefix else "?"
        return f"{base_path}{query_prefix}{sep}offset={new_offset}&limit={limit}"

    if offset > 0:
        _link(feed, rel="first", href=href(0), type=ACQ_TYPE)
        _link(feed, rel="prev", href=href(max(0, offset - limit)), type=ACQ_TYPE)
    if offset + limit < total:
        _link(feed, rel="next", href=href(offset + limit), type=ACQ_TYPE)
        last = ((total - 1) // limit) * limit if total > 0 else 0
        _link(feed, rel="last", href=href(last), type=ACQ_TYPE)


def _clamp_paging(offset: int, limit: int) -> tuple[int, int]:
    offset = max(0, offset)
    limit = max(1, min(limit, MAX_PAGE_SIZE))
    return offset, limit


def _render(feed: ET.Element, media_type: str) -> Response:
    body = b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(feed, encoding="utf-8")
    return Response(content=body, media_type=media_type)


# ---------- endpoints ---------------------------------------------------------


@router.get("", response_class=Response, include_in_schema=False)
@router.get("/", response_class=Response)
def opds_root() -> Response:
    feed = _feed_skeleton(
        title="Grimoire",
        feed_id="urn:grimoire:feed:root",
        self_path="/opds",
    )
    _nav_entry(
        feed,
        title="Recent additions",
        href="/opds/recent",
        summary="Newest items in the library.",
        kind=ACQ_TYPE,
    )
    _nav_entry(
        feed,
        title="Collections",
        href="/opds/collections",
        summary="Browse by collection.",
        kind=NAV_TYPE,
    )
    _nav_entry(
        feed,
        title="Tags",
        href="/opds/tags",
        summary="Browse by tag.",
        kind=NAV_TYPE,
    )
    _nav_entry(
        feed,
        title="Authors",
        href="/opds/authors",
        summary="Browse by author.",
        kind=NAV_TYPE,
    )
    _nav_entry(
        feed,
        title="By item type",
        href="/opds/types",
        summary="Browse by item type (paper, book, thesis, ...).",
        kind=NAV_TYPE,
    )
    _nav_entry(
        feed,
        title="Journals",
        href="/opds/venues",
        summary="Browse by journal / venue.",
        kind=NAV_TYPE,
    )
    _nav_entry(
        feed,
        title="Years",
        href="/opds/years",
        summary="Browse by publication year.",
        kind=NAV_TYPE,
    )
    return _render(feed, NAV_TYPE)


@router.get("/recent", response_class=Response)
def opds_recent(
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    offset, limit = _clamp_paging(offset, limit)
    items = queries.list_recent(conn, offset=offset, limit=limit)
    total = queries.count_all(conn)
    feed = _feed_skeleton(
        title="Recent additions",
        feed_id="urn:grimoire:feed:recent",
        self_path=f"/opds/recent?offset={offset}&limit={limit}",
    )
    for item in items:
        _acquisition_entry(feed, item)
    _paginate(feed, base_path="/opds/recent", offset=offset, limit=limit, total=total)
    return _render(feed, ACQ_TYPE)


@router.get("/collections", response_class=Response)
def opds_collections(
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    feed = _feed_skeleton(
        title="Collections",
        feed_id="urn:grimoire:feed:collections",
        self_path="/opds/collections",
    )
    for col in queries.list_collections(conn):
        if col.item_count == 0:
            continue
        _nav_entry(
            feed,
            title=col.name,
            href=f"/opds/collections/{col.collection_id}",
            summary=f"{col.item_count} item(s)",
            kind=ACQ_TYPE,
        )
    return _render(feed, NAV_TYPE)


@router.get("/collections/{collection_id}", response_class=Response)
def opds_collection(
    collection_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    col = queries.get_collection(conn, collection_id)
    if col is None:
        raise HTTPException(status_code=404, detail="collection not found")
    offset, limit = _clamp_paging(offset, limit)
    items = queries.list_in_collection(conn, collection_id, offset=offset, limit=limit)
    total = queries.count_in_collection(conn, collection_id)
    feed = _feed_skeleton(
        title=f"Collection: {col.name}",
        feed_id=f"urn:grimoire:feed:collection:{collection_id}",
        self_path=f"/opds/collections/{collection_id}?offset={offset}&limit={limit}",
    )
    for item in items:
        _acquisition_entry(feed, item)
    _paginate(
        feed,
        base_path=f"/opds/collections/{collection_id}",
        offset=offset,
        limit=limit,
        total=total,
    )
    return _render(feed, ACQ_TYPE)


@router.get("/tags", response_class=Response)
def opds_tags(
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    feed = _feed_skeleton(
        title="Tags",
        feed_id="urn:grimoire:feed:tags",
        self_path="/opds/tags",
    )
    for tag in queries.list_tags_with_counts(conn):
        if tag.item_count == 0:
            continue
        _nav_entry(
            feed,
            title=tag.name,
            href=f"/opds/tags/{tag.name}",
            summary=f"{tag.item_count} item(s)",
            kind=ACQ_TYPE,
        )
    return _render(feed, NAV_TYPE)


@router.get("/tags/{name}", response_class=Response)
def opds_tag(
    name: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    offset, limit = _clamp_paging(offset, limit)
    total = queries.count_with_tag(conn, name)
    if total == 0:
        # Tag not found (or empty) — 404 is more informative than an empty feed.
        tag_exists = conn.execute(
            "SELECT 1 FROM tags WHERE name = ?", (name,)
        ).fetchone()
        if tag_exists is None:
            raise HTTPException(status_code=404, detail="tag not found")
    items = queries.list_with_tag(conn, name, offset=offset, limit=limit)
    feed = _feed_skeleton(
        title=f"Tag: {name}",
        feed_id=f"urn:grimoire:feed:tag:{name}",
        self_path=f"/opds/tags/{name}?offset={offset}&limit={limit}",
    )
    for item in items:
        _acquisition_entry(feed, item)
    _paginate(feed, base_path=f"/opds/tags/{name}", offset=offset, limit=limit, total=total)
    return _render(feed, ACQ_TYPE)


@router.get("/authors", response_class=Response)
def opds_authors(
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    feed = _feed_skeleton(
        title="Authors",
        feed_id="urn:grimoire:feed:authors",
        self_path="/opds/authors",
    )
    for a in queries.list_authors(conn):
        display = (
            f"{a.given_name} {a.family_name}".strip() if a.given_name else a.family_name
        )
        _nav_entry(
            feed,
            title=display,
            href=f"/opds/authors/{a.author_id}",
            summary=f"{a.item_count} item(s)",
            kind=ACQ_TYPE,
        )
    return _render(feed, NAV_TYPE)


@router.get("/authors/{author_id}", response_class=Response)
def opds_author(
    author_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    a = queries.get_author(conn, author_id)
    if a is None:
        raise HTTPException(status_code=404, detail="author not found")
    offset, limit = _clamp_paging(offset, limit)
    items = queries.list_by_author(conn, author_id, offset=offset, limit=limit)
    total = queries.count_by_author(conn, author_id)
    display = f"{a.given_name} {a.family_name}".strip() if a.given_name else a.family_name
    feed = _feed_skeleton(
        title=f"Author: {display}",
        feed_id=f"urn:grimoire:feed:author:{author_id}",
        self_path=f"/opds/authors/{author_id}?offset={offset}&limit={limit}",
    )
    for item in items:
        _acquisition_entry(feed, item)
    _paginate(
        feed,
        base_path=f"/opds/authors/{author_id}",
        offset=offset,
        limit=limit,
        total=total,
    )
    return _render(feed, ACQ_TYPE)


@router.get("/types", response_class=Response)
def opds_types(
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    feed = _feed_skeleton(
        title="By item type",
        feed_id="urn:grimoire:feed:types",
        self_path="/opds/types",
    )
    for item_type in ITEM_TYPES:
        count = queries.count_by_type(conn, item_type)
        if count == 0:
            continue
        _nav_entry(
            feed,
            title=item_type.capitalize(),
            href=f"/opds/types/{item_type}",
            summary=f"{count} item(s)",
            kind=ACQ_TYPE,
        )
    return _render(feed, NAV_TYPE)


@router.get("/types/{item_type}", response_class=Response)
def opds_type(
    item_type: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    if item_type not in ITEM_TYPES:
        raise HTTPException(status_code=404, detail="unknown item type")
    offset, limit = _clamp_paging(offset, limit)
    items = queries.list_by_type(conn, item_type, offset=offset, limit=limit)
    total = queries.count_by_type(conn, item_type)
    feed = _feed_skeleton(
        title=f"Type: {item_type}",
        feed_id=f"urn:grimoire:feed:type:{item_type}",
        self_path=f"/opds/types/{item_type}?offset={offset}&limit={limit}",
    )
    for item in items:
        _acquisition_entry(feed, item)
    _paginate(
        feed,
        base_path=f"/opds/types/{item_type}",
        offset=offset,
        limit=limit,
        total=total,
    )
    return _render(feed, ACQ_TYPE)


@router.get("/search", response_class=Response)
def opds_search(
    q: str = Query(..., min_length=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    offset, limit = _clamp_paging(offset, limit)
    # OpenSearch: keyword-only (no embedder loading), paged in-memory since
    # FTS returns ranked hits and we can't efficiently OFFSET through them.
    hits = keyword_search.search_items(conn, q, limit=offset + limit)
    paged_ids = [h.item_id for h in hits[offset : offset + limit]]
    items = queries.hydrate_by_ids(conn, paged_ids)
    feed = _feed_skeleton(
        title=f"Search: {q}",
        feed_id=f"urn:grimoire:feed:search:{q}",
        self_path=f"/opds/search?q={q}&offset={offset}&limit={limit}",
    )
    _sub(feed, _qname(OPENSEARCH_NS, "totalResults"), str(len(hits)))
    _sub(feed, _qname(OPENSEARCH_NS, "startIndex"), str(offset + 1))
    _sub(feed, _qname(OPENSEARCH_NS, "itemsPerPage"), str(limit))
    for item in items:
        _acquisition_entry(feed, item)
    # We only know the size of the current FTS cap, not the full result set,
    # so skip last-link; prev/next still work for the typical browse case.
    if offset > 0:
        _link(
            feed,
            rel="prev",
            href=f"/opds/search?q={q}&offset={max(0, offset - limit)}&limit={limit}",
            type=ACQ_TYPE,
        )
    if len(hits) > offset + limit:
        _link(
            feed,
            rel="next",
            href=f"/opds/search?q={q}&offset={offset + limit}&limit={limit}",
            type=ACQ_TYPE,
        )
    return _render(feed, ACQ_TYPE)


@router.get("/venues", response_class=Response)
def opds_venues(
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    from urllib.parse import quote

    feed = _feed_skeleton(
        title="Journals",
        feed_id="urn:grimoire:feed:venues",
        self_path="/opds/venues",
    )
    for v in queries.list_venues_with_counts(conn):
        if v.item_count == 0:
            continue
        _nav_entry(
            feed,
            title=v.venue,
            href=f"/opds/venues/{quote(v.venue, safe='')}",
            summary=f"{v.item_count} item(s)",
            kind=ACQ_TYPE,
        )
    return _render(feed, NAV_TYPE)


@router.get("/venues/{venue:path}", response_class=Response)
def opds_venue(
    venue: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    from urllib.parse import quote

    offset, limit = _clamp_paging(offset, limit)
    total = queries.count_by_venue(conn, venue)
    if total == 0:
        raise HTTPException(status_code=404, detail="venue not found")
    items = queries.list_by_venue(conn, venue, offset=offset, limit=limit)
    base = f"/opds/venues/{quote(venue, safe='')}"
    feed = _feed_skeleton(
        title=f"Journal: {venue}",
        feed_id=f"urn:grimoire:feed:venue:{venue}",
        self_path=f"{base}?offset={offset}&limit={limit}",
    )
    for item in items:
        _acquisition_entry(feed, item)
    _paginate(feed, base_path=base, offset=offset, limit=limit, total=total)
    return _render(feed, ACQ_TYPE)


@router.get("/years", response_class=Response)
def opds_years(
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    feed = _feed_skeleton(
        title="Years",
        feed_id="urn:grimoire:feed:years",
        self_path="/opds/years",
    )
    for y in queries.list_years_with_counts(conn):
        _nav_entry(
            feed,
            title=str(y.year),
            href=f"/opds/years/{y.year}",
            summary=f"{y.item_count} item(s)",
            kind=ACQ_TYPE,
        )
    return _render(feed, NAV_TYPE)


@router.get("/years/{year}", response_class=Response)
def opds_year(
    year: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    conn: sqlite3.Connection = Depends(_db),
) -> Response:
    offset, limit = _clamp_paging(offset, limit)
    total = queries.count_by_year(conn, year)
    if total == 0:
        raise HTTPException(status_code=404, detail="year not found")
    items = queries.list_by_year(conn, year, offset=offset, limit=limit)
    feed = _feed_skeleton(
        title=f"Year: {year}",
        feed_id=f"urn:grimoire:feed:year:{year}",
        self_path=f"/opds/years/{year}?offset={offset}&limit={limit}",
    )
    for item in items:
        _acquisition_entry(feed, item)
    _paginate(
        feed, base_path=f"/opds/years/{year}", offset=offset, limit=limit, total=total
    )
    return _render(feed, ACQ_TYPE)


@router.get("/opensearch.xml", response_class=Response)
def opensearch_description() -> Response:
    """OpenSearch 1.1 description document. Clients fetch this to learn how
    to form a query URL for /opds/search."""
    root = ET.Element(_qname(OPENSEARCH_NS, "OpenSearchDescription"))
    _sub(root, _qname(OPENSEARCH_NS, "ShortName"), "Grimoire")
    _sub(root, _qname(OPENSEARCH_NS, "Description"), "Search the Grimoire library")
    _sub(root, _qname(OPENSEARCH_NS, "InputEncoding"), "UTF-8")
    url = ET.SubElement(root, _qname(OPENSEARCH_NS, "Url"))
    url.set("type", ACQ_TYPE)
    url.set("template", "/opds/search?q={searchTerms}")
    return _render(root, OPENSEARCH_TYPE)
