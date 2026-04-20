"""FastMCP server wiring. Tool names/signatures match plan §6 Phase 4.

The actual logic lives in ``grimoire.mcp.tools``; these wrappers open a fresh
SQLite connection per call (MCP tool calls can arrive concurrently), lazily
load embedders, and serialize pydantic models to plain dicts for the protocol."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP

from grimoire.db import apply_migrations, connect
from grimoire.mcp import tools as tools_impl
from grimoire.mcp.citation import to_bibtex
from grimoire.mcp.models import Collection, ItemFull, ItemSummary, RelatedItem, Snippet

log = logging.getLogger(__name__)

mcp = FastMCP(
    "grimoire",
    instructions=(
        "Grimoire: a self-hosted literature library. Tools let you search "
        "papers/books/reports by keyword or semantic similarity, pull full "
        "text by page, list related works (preprints, errata, follow-ups), "
        "and export BibTeX citations."
    ),
    # We mount this ASGI sub-app at /mcp in FastAPI, so the sub-app's own
    # path is root-relative. Otherwise the final URL would be /mcp/mcp.
    streamable_http_path="/",
)


@contextlib.contextmanager
def _db() -> Iterator[Any]:
    """Per-call sqlite connection. Cheap on local SQLite; avoids thread issues."""
    conn = connect()
    apply_migrations(conn)
    try:
        yield conn
    finally:
        conn.close()


@lru_cache(maxsize=1)
def _item_embedder() -> Any:
    from grimoire.embed.specter2 import Specter2Embedder

    return Specter2Embedder()


@lru_cache(maxsize=1)
def _chunk_embedder() -> Any:
    from grimoire.embed.bge_m3 import BGEM3Embedder

    return BGEM3Embedder()


# ---------- tools ---------------------------------------------------------


@mcp.tool()
def search(
    query: str,
    mode: str = "hybrid",
    item_types: list[str] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search the library. Modes: 'keyword' (FTS5), 'semantic' (SPECTER2 KNN),
    'hybrid' (RRF fusion — best default). ``item_types`` filters to e.g.
    ['paper', 'book', 'thesis']. Returns ranked items, each with a best-matching
    chunk snippet and page number when available."""
    if mode not in {"hybrid", "keyword", "semantic"}:
        raise ValueError(f"mode must be hybrid|keyword|semantic, got {mode!r}")
    item_emb = _item_embedder() if mode in {"hybrid", "semantic"} else None
    chunk_emb = _chunk_embedder() if mode in {"hybrid", "semantic"} else None
    with _db() as conn:
        hits = tools_impl.search(
            conn,
            query,
            mode=mode,  # type: ignore[arg-type]
            item_types=item_types,
            limit=limit,
            item_embedder=item_emb,
            chunk_embedder=chunk_emb,
        )
    return [_dump(h) for h in hits]


@mcp.tool()
def get_item(item_id: int) -> dict[str, Any] | None:
    """Get full metadata for a single item (title, abstract, authors, venue,
    volume/issue/pages, identifiers, tags, collections)."""
    with _db() as conn:
        item = tools_impl.get_item(conn, item_id)
    return _dump(item) if item else None


@mcp.tool()
def get_full_text(item_id: int, page: int | None = None) -> str:
    """Return the body text of an item, reconstructed from its indexed chunks.
    When ``page`` is supplied (1-indexed), return only chunks tagged with that
    page number. Empty string for items with no indexed body."""
    with _db() as conn:
        return tools_impl.get_full_text(conn, item_id, page)


@mcp.tool()
def get_snippets(query: str, item_id: int | None = None, k: int = 10) -> list[dict[str, Any]]:
    """Best-matching chunks for a query. If ``item_id`` is set, snippets are
    restricted to that one item. Uses BGE-M3 semantic search; falls back to
    FTS5 when the chunk embedder is unavailable."""
    with _db() as conn:
        snippets = tools_impl.get_snippets(
            conn, query, item_id=item_id, k=k, chunk_embedder=_chunk_embedder()
        )
    return [_dump(s) for s in snippets]


@mcp.tool()
def list_related(item_id: int, kind: str = "all") -> list[dict[str, Any]]:
    """List related items. ``kind`` ∈ {all, preprint_chain, semantic, citations}.
    preprint_chain covers preprint↔published and edition relations; semantic
    is the 'related' relation surfaced by tier-4 dedup; citations covers
    explicit cite/cited_by."""
    if kind not in {"all", "preprint_chain", "semantic", "citations"}:
        raise ValueError("kind must be one of all|preprint_chain|semantic|citations")
    with _db() as conn:
        rel = tools_impl.list_related(conn, item_id, kind)  # type: ignore[arg-type]
    return [_dump(r) for r in rel]


@mcp.tool()
def get_citation(item_id: int, style: str = "bibtex") -> str:
    """Generate a citation string. ``style`` is currently only 'bibtex'; CSL
    styles (APA/IEEE/Nature) are deferred to v2."""
    if style != "bibtex":
        raise ValueError("only 'bibtex' style is supported in v1")
    with _db() as conn:
        bib = to_bibtex(conn, item_id)
    if bib is None:
        raise ValueError(f"item {item_id} not found")
    return bib


@mcp.tool()
def list_tags() -> list[str]:
    """Return every tag in the library, alphabetical."""
    with _db() as conn:
        return tools_impl.list_tags(conn)


@mcp.tool()
def list_collections() -> list[dict[str, Any]]:
    """Return collections with item counts. Each row has id, name, parent_id
    so the client can reconstruct the tree."""
    with _db() as conn:
        return [_dump(c) for c in tools_impl.list_collections(conn)]


@mcp.tool()
def find_by_tag(tag: str, limit: int = 100) -> list[dict[str, Any]]:
    """List items carrying a given tag, newest first."""
    with _db() as conn:
        items = tools_impl.find_by_tag(conn, tag, limit=limit)
    return [_dump(i) for i in items]


def _dump(model: ItemSummary | ItemFull | Snippet | RelatedItem | Collection) -> dict[str, Any]:
    return model.model_dump(exclude_none=False)
