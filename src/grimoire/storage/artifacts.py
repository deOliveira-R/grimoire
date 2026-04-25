"""Per-item derived artifacts tracked via item_artifacts.

Each item can carry several files:
  * 'primary'      - the original PDF / EPUB (mirrors items.content_hash)
  * 'grobid_tei'   - GROBID TEI XML (structured sections + references)
  * 'ocr_text'     - OCR plain text (for scanned PDFs, v2)
  * 'extracted_md' - reduced markdown cache (v2)

All blobs land in the same CAS. This module is the single read/write point —
nothing else should poke at ``item_artifacts`` directly."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from grimoire.config import settings
from grimoire.storage.cas import CAS

Kind = Literal["primary", "grobid_tei", "ocr_text", "extracted_md"]


@dataclass(slots=True)
class ArtifactRef:
    item_id: int
    kind: str
    content_hash: str
    source: str | None
    generated_at: str
    size_bytes: int | None


def store(
    conn: sqlite3.Connection,
    item_id: int,
    kind: Kind,
    data: bytes,
    *,
    source: str | None = None,
) -> str:
    """Write ``data`` to CAS, upsert the (item, kind) row. Returns the hash.

    Re-running with different bytes replaces the mapping — the previous blob
    stays in CAS (orphan, GC'd later — we have no GC today). Re-running with
    identical bytes is a no-op on disk (CAS dedups by hash)."""
    cas = CAS(settings.files_root)
    content_hash, _ = cas.store(data)
    conn.execute(
        """INSERT INTO item_artifacts(item_id, kind, content_hash, source, size_bytes)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(item_id, kind) DO UPDATE SET
             content_hash = excluded.content_hash,
             source       = excluded.source,
             generated_at = CURRENT_TIMESTAMP,
             size_bytes   = excluded.size_bytes""",
        (item_id, kind, content_hash, source, len(data)),
    )
    return content_hash


def register(
    conn: sqlite3.Connection,
    item_id: int,
    kind: Kind,
    content_hash: str,
    *,
    source: str | None = None,
    size_bytes: int | None = None,
) -> None:
    """Record an (item, kind) → existing CAS hash mapping without re-storing.

    Use when the bytes are already in CAS (e.g. ``cas.store_file`` ran earlier
    in the same transaction) and we just need the row in ``item_artifacts``.
    Idempotent: re-runs replace the row in place, mirroring ``store``."""
    conn.execute(
        """INSERT INTO item_artifacts(item_id, kind, content_hash, source, size_bytes)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(item_id, kind) DO UPDATE SET
             content_hash = excluded.content_hash,
             source       = excluded.source,
             generated_at = CURRENT_TIMESTAMP,
             size_bytes   = excluded.size_bytes""",
        (item_id, kind, content_hash, source, size_bytes),
    )


def get_hash(
    conn: sqlite3.Connection, item_id: int, kind: Kind
) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM item_artifacts WHERE item_id = ? AND kind = ?",
        (item_id, kind),
    ).fetchone()
    return row["content_hash"] if row else None


def path_for(
    conn: sqlite3.Connection, item_id: int, kind: Kind
) -> Path | None:
    """Return the on-disk CAS path for an artifact, or ``None`` if either the
    DB row or the file itself is missing."""
    h = get_hash(conn, item_id, kind)
    if h is None:
        return None
    cas = CAS(settings.files_root)
    p = cas.path_for_hash(h)
    return p if p.exists() else None


def read(
    conn: sqlite3.Connection, item_id: int, kind: Kind
) -> bytes | None:
    p = path_for(conn, item_id, kind)
    return p.read_bytes() if p is not None else None


def exists(conn: sqlite3.Connection, item_id: int, kind: Kind) -> bool:
    return path_for(conn, item_id, kind) is not None


def info(
    conn: sqlite3.Connection, item_id: int, kind: Kind
) -> ArtifactRef | None:
    row = conn.execute(
        "SELECT item_id, kind, content_hash, source, generated_at, size_bytes "
        "FROM item_artifacts WHERE item_id = ? AND kind = ?",
        (item_id, kind),
    ).fetchone()
    if row is None:
        return None
    return ArtifactRef(
        item_id=int(row["item_id"]),
        kind=row["kind"],
        content_hash=row["content_hash"],
        source=row["source"],
        generated_at=row["generated_at"],
        size_bytes=row["size_bytes"],
    )


def items_missing_kind(
    conn: sqlite3.Connection,
    kind: Kind,
    *,
    primary_only: bool = True,
    limit: int | None = None,
) -> list[int]:
    """Return ids of items that have a 'primary' artifact but no artifact of
    the requested kind. ``primary_only=False`` drops the primary-present
    prerequisite (useful if a kind is derived from something other than the
    canonical PDF)."""
    if primary_only:
        sql = """
            SELECT ap.item_id
            FROM item_artifacts ap
            LEFT JOIN item_artifacts ak
              ON ak.item_id = ap.item_id AND ak.kind = ?
            WHERE ap.kind = 'primary' AND ak.item_id IS NULL
            ORDER BY ap.item_id
        """
        params: tuple[object, ...] = (kind,)
    else:
        sql = """
            SELECT i.id AS item_id
            FROM items i
            LEFT JOIN item_artifacts a
              ON a.item_id = i.id AND a.kind = ?
            WHERE a.item_id IS NULL
            ORDER BY i.id
        """
        params = (kind,)
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [int(r["item_id"]) for r in conn.execute(sql, params).fetchall()]
