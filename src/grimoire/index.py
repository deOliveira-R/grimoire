"""Backfill / incremental indexing pipeline.

Given an already-ingested item (row in ``items`` with content_hash → CAS file),
produce:
  * one SPECTER2 item-level embedding (title + [SEP] + abstract; body fallback)
  * N chunks with BGE-M3 chunk-level embeddings

Both embedders must be injectable so tests can run without downloading the
real models."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from grimoire.chunk import Chunk, chunk_pages
from grimoire.config import settings
from grimoire.embed.base import Embedder, l2_normalize, serialize_float32
from grimoire.embed.specter2 import format_item_text
from grimoire.storage.cas import CAS

log = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexResult:
    item_id: int
    status: Literal["indexed", "skipped", "failed"]
    chunks: int = 0
    reason: str | None = None


def index_item(
    conn: sqlite3.Connection,
    item_id: int,
    *,
    item_embedder: Embedder,
    chunk_embedder: Embedder,
    force: bool = False,
) -> IndexResult:
    row = conn.execute(
        "SELECT title, abstract, content_hash FROM items WHERE id=?", (item_id,)
    ).fetchone()
    if row is None:
        return IndexResult(item_id=item_id, status="failed", reason="item not found")

    has_item_emb = (
        conn.execute("SELECT 1 FROM item_embeddings WHERE item_id=?", (item_id,)).fetchone()
        is not None
    )
    if has_item_emb and not force:
        return IndexResult(item_id=item_id, status="skipped", reason="already indexed")

    # Pages = per-page text from the primary file if we have one; else empty.
    pages = _extract_pages(row["content_hash"])
    body_text = "\n\n".join(text for _, text in pages) if pages else None

    # Item embedding — SPECTER2 expects "title [SEP] abstract".
    item_text = format_item_text(
        title=row["title"] or "",
        abstract=row["abstract"],
        body_fallback=body_text,
    )
    item_vec = l2_normalize(item_embedder.encode([item_text]))[0]
    _dim_check(item_vec, item_embedder.dim, "item")

    if force:
        conn.execute("DELETE FROM item_embeddings WHERE item_id=?", (item_id,))
        # chunk_embeddings is a vec0 virtual table with no FK cascade —
        # clear its rows explicitly before we drop the chunks themselves.
        for (cid,) in conn.execute("SELECT id FROM chunks WHERE item_id=?", (item_id,)):
            conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id=?", (cid,))
        conn.execute("DELETE FROM chunks WHERE item_id=?", (item_id,))
    conn.execute(
        "INSERT INTO item_embeddings(item_id, embedding) VALUES (?, ?)",
        (item_id, serialize_float32(item_vec)),
    )

    # Chunk embeddings — only meaningful for items with a body.
    n_chunks = 0
    if pages:
        chunks = chunk_pages(pages)
        if chunks:
            n_chunks = _insert_chunks_with_embeddings(conn, item_id, chunks, chunk_embedder)

    return IndexResult(item_id=item_id, status="indexed", chunks=n_chunks)


def index_all(
    conn: sqlite3.Connection,
    *,
    item_embedder: Embedder,
    chunk_embedder: Embedder,
    force: bool = False,
    limit: int | None = None,
) -> list[IndexResult]:
    q = "SELECT id FROM items ORDER BY id"
    if limit is not None:
        q += f" LIMIT {int(limit)}"
    ids = [row["id"] for row in conn.execute(q).fetchall()]
    return [
        index_item(
            conn,
            item_id,
            item_embedder=item_embedder,
            chunk_embedder=chunk_embedder,
            force=force,
        )
        for item_id in ids
    ]


# ---------- internals ------------------------------------------------------


def _extract_pages(content_hash: str | None) -> list[tuple[int, str]]:
    if not content_hash:
        return []
    cas = CAS(settings.files_root)
    path = cas.path_for_hash(content_hash)
    if not path.exists():
        return []
    kind = _detect_kind(path)
    if kind == "pdf":
        return _pdf_pages(path)
    if kind == "epub":
        from grimoire.extract import epub as epub_extract

        # EPUBs have no true page concept — collapse to page 1.
        text = epub_extract.extract_text(path)
        return [(1, text)] if text.strip() else []
    return []


def _detect_kind(path: Path) -> Literal["pdf", "epub", "unknown"]:
    """File extensions are stripped by the CAS layout; detect via magic bytes."""
    with path.open("rb") as fh:
        head = fh.read(4)
    if head[:4] == b"%PDF":
        return "pdf"
    if head[:2] == b"PK":  # zip magic — EPUB and docx etc. For our inputs, EPUB.
        return "epub"
    return "unknown"


def _pdf_pages(path: Path) -> list[tuple[int, str]]:
    import pymupdf

    out: list[tuple[int, str]] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if text and text.strip():
                out.append((i, text))
    return out


def _insert_chunks_with_embeddings(
    conn: sqlite3.Connection,
    item_id: int,
    chunks: list[Chunk],
    chunk_embedder: Embedder,
) -> int:
    texts = [c.text for c in chunks]
    vecs = l2_normalize(chunk_embedder.encode(texts))
    _dim_check(vecs, chunk_embedder.dim, "chunk")

    for chunk, vec in zip(chunks, vecs, strict=True):
        cur = conn.execute(
            "INSERT INTO chunks(item_id, chunk_index, page, text) VALUES (?, ?, ?, ?)",
            (item_id, chunk.chunk_index, chunk.page, chunk.text),
        )
        chunk_id = int(cur.lastrowid)  # type: ignore[arg-type]
        conn.execute(
            "INSERT INTO chunk_embeddings(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_float32(vec)),
        )
    return len(chunks)


def _dim_check(arr: object, expected: int, label: str) -> None:
    import numpy as np

    a = arr if isinstance(arr, np.ndarray) else np.asarray(arr)
    got = a.shape[-1]
    if got != expected:
        raise ValueError(f"{label} embedding dim mismatch: expected {expected}, got {got}")


def walk_unindexed(conn: sqlite3.Connection) -> Iterable[int]:
    for row in conn.execute(
        """SELECT i.id
           FROM items i
           LEFT JOIN item_embeddings e ON e.item_id = i.id
           WHERE e.item_id IS NULL
           ORDER BY i.id"""
    ):
        yield int(row["id"])
