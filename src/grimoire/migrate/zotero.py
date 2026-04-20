"""Migrate a local Zotero SQLite library into Grimoire (plan §6 Phase 7).

Reads ``zotero.sqlite`` read-only and pumps top-level items into the
grimoire DB, preserving:
  * bibliographic metadata (title, abstract, year, DOI, ISBN, venue,
    volume, issue, pages, series, edition, language)
  * authors + creator roles (author, editor, translator — mapped to the
    grimoire ``item_authors.role`` vocabulary)
  * tags
  * collections (including nested hierarchy)
  * PDF attachments: CAS-stored and linked to the item

Idempotent via ``metadata_json.zotero_item_id`` — re-running the migration
only touches items not already imported.

Dedup still runs (tier 1 deterministic) so Zotero-internal duplicates
(same DOI on two Zotero entries) collapse. Tier 4 is opt-in via
``item_embedder`` for a one-shot semantic sweep at the end, but the
migration itself keeps the default off to stay fast."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grimoire import dedup
from grimoire.config import settings
from grimoire.dedup import JudgeFn
from grimoire.embed.base import Embedder
from grimoire.models import Author, Metadata
from grimoire.storage.cas import CAS

log = logging.getLogger(__name__)


# Zotero itemTypes → Grimoire item_type vocabulary.
_TYPE_MAP: dict[str, str] = {
    "journalArticle": "paper",
    "conferencePaper": "paper",
    "preprint": "preprint",
    "book": "book",
    "bookSection": "chapter",
    "report": "report",
    "thesis": "thesis",
    "standard": "standard",
    "patent": "patent",
    "magazineArticle": "paper",
    "newspaperArticle": "paper",
    "manuscript": "other",
    "webpage": "other",
    "document": "other",
    "letter": "other",
    "presentation": "other",
    "computerProgram": "other",
}

# Zotero creator roles → Grimoire ``item_authors.role`` (plan schema).
_ROLE_MAP: dict[str, str] = {
    "author": "author",
    "bookAuthor": "author",
    "contributor": "author",
    "reviewedAuthor": "author",
    "inventor": "author",
    "programmer": "author",
    "originalCreator": "author",
    "creator": "author",
    "editor": "editor",
    "seriesEditor": "editor",
    "translator": "translator",
}


@dataclass(slots=True)
class MigrationReport:
    total_candidates: int = 0
    inserted: int = 0
    merged: int = 0  # tier-1 identifier collision with an already-imported item
    skipped_already_imported: int = 0
    skipped_no_metadata: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    pdf_attachments_stored: int = 0


# ---------- public API ------------------------------------------------------


def migrate(
    conn: sqlite3.Connection,
    *,
    library_path: Path,
    storage_dir: Path,
    limit: int | None = None,
    dry_run: bool = False,
    item_embedder: Embedder | None = None,
    llm_judge: JudgeFn | None = None,
) -> MigrationReport:
    """Copy top-level items from the Zotero SQLite at ``library_path`` into
    the open grimoire ``conn``. ``storage_dir`` is the Zotero ``storage/``
    directory where PDF attachments live.

    Any work done by this function is inside a single, explicit BEGIN/COMMIT
    so a Ctrl-C halfway leaves the grimoire DB untouched."""
    if not library_path.exists():
        raise FileNotFoundError(f"Zotero SQLite not found: {library_path}")

    zotero = _open_readonly(library_path)
    try:
        candidates = _fetch_items(zotero, limit=limit)
        report = MigrationReport(total_candidates=len(candidates))
        if dry_run:
            # Just describe what we'd do; don't touch the target DB.
            log.info("dry-run: %d candidates ready to import", len(candidates))
            return report

        cas = CAS(settings.files_root)
        conn.execute("BEGIN")
        try:
            for zi in candidates:
                _import_one(conn, zotero, zi, storage_dir, cas, report, item_embedder, llm_judge)
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        zotero.close()

    return report


# ---------- Zotero SQLite reader --------------------------------------------


def _open_readonly(path: Path) -> sqlite3.Connection:
    """Open the Zotero DB in readonly+immutable mode so a running Zotero
    client can't block us and we can't accidentally mutate it.

    SQLite URIs need url-quoting for anything non-alphanumeric. Rodrigo's
    library path contains ``#Zotero`` which would otherwise terminate the
    URI at the fragment separator — use ``pathlib.Path.as_uri()`` semantics
    manually."""
    from urllib.parse import quote

    uri = f"file:{quote(str(path))}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@dataclass(slots=True)
class _ZoteroItem:
    item_id: int
    type_name: str


def _fetch_items(zotero: sqlite3.Connection, *, limit: int | None) -> list[_ZoteroItem]:
    """Top-level items only: skip attachments (they're metadata about files,
    not bibliographic records) and anything in the trash."""
    excluded = "('attachment', 'note', 'annotation')"
    sql = f"""
        SELECT i.itemID, it.typeName
        FROM items i
        JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
        WHERE it.typeName NOT IN {excluded}
          AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
        ORDER BY i.itemID
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [
        _ZoteroItem(item_id=int(r["itemID"]), type_name=r["typeName"])
        for r in zotero.execute(sql).fetchall()
    ]


# Cache field IDs across all item reads — tiny table, queried thousands of times.
_FIELD_CACHE: dict[int, dict[str, int]] = {}


def _field_ids(zotero: sqlite3.Connection) -> dict[str, int]:
    key = id(zotero)
    if key not in _FIELD_CACHE:
        rows = zotero.execute("SELECT fieldName, fieldID FROM fields").fetchall()
        _FIELD_CACHE[key] = {r["fieldName"]: int(r["fieldID"]) for r in rows}
    return _FIELD_CACHE[key]


def _field_value(zotero: sqlite3.Connection, item_id: int, field_id: int) -> str | None:
    row = zotero.execute(
        """SELECT v.value FROM itemData d
           JOIN itemDataValues v ON v.valueID = d.valueID
           WHERE d.itemID = ? AND d.fieldID = ?""",
        (item_id, field_id),
    ).fetchone()
    return row["value"] if row else None


def _year_from_date(date: str | None) -> int | None:
    if not date:
        return None
    head = date.strip()[:4]
    if head.isdigit() and 1000 <= int(head) <= 2100:
        return int(head)
    for tok in (date or "").split():
        if tok.isdigit() and 1000 <= int(tok) <= 2100:
            return int(tok)
    return None


def _creators(zotero: sqlite3.Connection, item_id: int) -> list[tuple[Author, str]]:
    """Return ``[(Author, role), ...]`` where ``role`` is the grimoire role
    vocabulary. Zotero roles not in ``_ROLE_MAP`` are dropped — they're
    usually minor contributors (commenter, sponsor, cast) that don't belong
    in the bibliographic record."""
    rows = zotero.execute(
        """SELECT c.lastName, c.firstName, ct.creatorType
           FROM itemCreators ic
           JOIN creators c ON c.creatorID = ic.creatorID
           JOIN creatorTypes ct ON ct.creatorTypeID = ic.creatorTypeID
           WHERE ic.itemID = ?
           ORDER BY ic.orderIndex""",
        (item_id,),
    ).fetchall()
    out: list[tuple[Author, str]] = []
    for r in rows:
        last = (r["lastName"] or "").strip()
        if not last:
            # Zotero supports single-field "institution" names (only firstName).
            # Keep them if present; grimoire stores institution in family_name.
            last = (r["firstName"] or "").strip()
            first = None
        else:
            first = (r["firstName"] or "").strip() or None
        if not last:
            continue
        role = _ROLE_MAP.get(r["creatorType"])
        if role is None:
            continue
        out.append((Author(family_name=last, given_name=first), role))
    return out


def _tags(zotero: sqlite3.Connection, item_id: int) -> list[str]:
    return [
        r["name"]
        for r in zotero.execute(
            """SELECT t.name FROM itemTags it
               JOIN tags t ON t.tagID = it.tagID
               WHERE it.itemID = ?""",
            (item_id,),
        ).fetchall()
    ]


def _collections(zotero: sqlite3.Connection, item_id: int) -> list[tuple[int, str, int | None]]:
    """Return ``[(zotero_collection_id, name, zotero_parent_collection_id), ...]``.

    Parent IDs are Zotero-space — they need remapping against the grimoire
    collections table at write time."""
    rows = zotero.execute(
        """SELECT c.collectionID, c.collectionName, c.parentCollectionID
           FROM collectionItems ci
           JOIN collections c ON c.collectionID = ci.collectionID
           WHERE ci.itemID = ?""",
        (item_id,),
    ).fetchall()
    return [
        (int(r["collectionID"]), r["collectionName"], r["parentCollectionID"])
        for r in rows
    ]


def _pdf_attachment(
    zotero: sqlite3.Connection, item_id: int, storage_dir: Path
) -> Path | None:
    """Return the first readable PDF attachment on disk, or ``None``."""
    rows = zotero.execute(
        """SELECT it.key, a.path
           FROM itemAttachments a
           JOIN items it ON it.itemID = a.itemID
           WHERE a.parentItemID = ? AND a.contentType = 'application/pdf'
           ORDER BY it.itemID""",
        (item_id,),
    ).fetchall()
    for r in rows:
        path: str = r["path"] or ""
        key: str = r["key"] or ""
        if not path:
            continue
        if path.startswith("storage:"):
            candidate: Path = storage_dir / key / path[len("storage:") :]
            if candidate.exists():
                return candidate
        elif path.startswith("attachments:"):
            # Absolute / relative-to-linked-attachments — we don't resolve these.
            continue
        else:
            candidate = Path(path)
            if candidate.exists():
                return candidate
    return None


# ---------- metadata assembly ----------------------------------------------


def _build_metadata(zotero: sqlite3.Connection, zi: _ZoteroItem) -> Metadata | None:
    fid = _field_ids(zotero)

    def get(name: str) -> str | None:
        field_id = fid.get(name)
        return _field_value(zotero, zi.item_id, field_id) if field_id is not None else None

    title = get("title")
    if not title:
        return None  # Zotero entries without a title aren't bibliographic items.

    authors_with_roles = _creators(zotero, zi.item_id)

    return Metadata(
        title=title,
        abstract=get("abstractNote"),
        publication_year=_year_from_date(get("date")),
        doi=get("DOI"),
        isbn=get("ISBN"),
        venue=get("publicationTitle"),
        volume=get("volume"),
        issue=get("issue"),
        pages=get("pages"),
        series=get("series"),
        series_number=get("seriesNumber"),
        edition=get("edition"),
        language=get("language"),
        item_type=_TYPE_MAP.get(zi.type_name, "other"),
        authors=[a for a, _ in authors_with_roles],  # roles handled below on insert
        source="zotero_import",
        confidence=0.9,
        raw={"zotero_item_id": zi.item_id, "zotero_type": zi.type_name},
    )


# ---------- importing ------------------------------------------------------


def _import_one(
    conn: sqlite3.Connection,
    zotero: sqlite3.Connection,
    zi: _ZoteroItem,
    storage_dir: Path,
    cas: CAS,
    report: MigrationReport,
    item_embedder: Embedder | None,
    llm_judge: JudgeFn | None,
) -> None:
    # Idempotency: if we've already imported this Zotero id, skip.
    existing = conn.execute(
        "SELECT id FROM items WHERE json_extract(metadata_json, '$.zotero_item_id') = ?",
        (zi.item_id,),
    ).fetchone()
    if existing:
        report.skipped_already_imported += 1
        return

    metadata = _build_metadata(zotero, zi)
    if metadata is None:
        report.skipped_no_metadata += 1
        return

    # Hash the PDF (if any) so tier-1 hash matching works against papers that
    # were ingested earlier through the CLI.
    pdf_path = _pdf_attachment(zotero, zi.item_id, storage_dir)
    content_hash: str | None = None
    if pdf_path is not None:
        try:
            content_hash = CAS.hash_file(pdf_path)
        except OSError as exc:
            report.failures.append(
                {"zotero_item_id": zi.item_id, "reason": f"hash failed: {exc}"}
            )
            pdf_path = None

    decision = dedup.decide(
        conn,
        metadata,
        content_hash,
        item_embedder=item_embedder,
        llm_judge=llm_judge,
    )

    if decision.outcome == "merge":
        target = decision.target_id
        assert target is not None
        if pdf_path is not None:
            cas.store_file(pdf_path)
            report.pdf_attachments_stored += 1
        dedup.apply_merge(conn, target, metadata)
        # Persist Zotero id on the merged row so re-runs skip correctly.
        _stamp_zotero_id(conn, target, zi.item_id)
        _record(conn, f"<zotero:{zi.item_id}>", content_hash, "merged", target)
        report.merged += 1
        return

    # decision is insert or link — in both cases we create a row.
    if pdf_path is not None:
        cas.store_file(pdf_path)
        report.pdf_attachments_stored += 1

    item_id = _insert_item(conn, metadata, content_hash)
    _insert_authors_with_roles(conn, item_id, _creators(zotero, zi.item_id))
    _apply_tags(conn, item_id, _tags(zotero, zi.item_id))
    _apply_collections(conn, item_id, _collections(zotero, zi.item_id))

    if decision.outcome == "link":
        target = decision.target_id
        relation = decision.relation
        assert target is not None and relation is not None
        dedup.apply_link(conn, item_id, target, relation, decision.confidence)

    _record(conn, f"<zotero:{zi.item_id}>", content_hash, "inserted", item_id)
    report.inserted += 1


def _insert_item(
    conn: sqlite3.Connection, md: Metadata, content_hash: str | None
) -> int:
    cur = conn.execute(
        """INSERT INTO items(
            item_type, title, abstract, publication_year,
            doi, arxiv_id, isbn,
            venue, volume, issue, pages, series, series_number, edition, language,
            content_hash, file_path,
            metadata_source, metadata_confidence, metadata_json
        ) VALUES (?,?,?,?, ?,?,?, ?,?,?,?,?,?,?,?, ?,?, ?,?,?)""",
        (
            md.item_type,
            md.title or "(untitled)",
            md.abstract,
            md.publication_year,
            md.doi,
            md.arxiv_id,
            md.isbn,
            md.venue,
            md.volume,
            md.issue,
            md.pages,
            md.series,
            md.series_number,
            md.edition,
            md.language,
            content_hash,
            _cas_relpath(content_hash) if content_hash else None,
            md.source,
            md.confidence,
            json.dumps(md.raw) if md.raw is not None else None,
        ),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def _cas_relpath(h: str) -> str:
    return f"{h[:2]}/{h[2:4]}/{h}"


def _insert_authors_with_roles(
    conn: sqlite3.Connection,
    item_id: int,
    creators: list[tuple[Author, str]],
) -> None:
    for pos, (author, role) in enumerate(creators):
        author_id = _upsert_author(conn, author)
        conn.execute(
            """INSERT OR IGNORE INTO item_authors(item_id, author_id, position, role)
               VALUES (?, ?, ?, ?)""",
            (item_id, author_id, pos, role),
        )


def _upsert_author(conn: sqlite3.Connection, author: Author) -> int:
    if author.orcid:
        row = conn.execute("SELECT id FROM authors WHERE orcid = ?", (author.orcid,)).fetchone()
        if row:
            return int(row["id"])
    row = conn.execute(
        "SELECT id FROM authors WHERE normalized_key = ? AND orcid IS ?",
        (author.normalized_key, author.orcid),
    ).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO authors(family_name, given_name, orcid, normalized_key) "
        "VALUES (?, ?, ?, ?)",
        (author.family_name, author.given_name, author.orcid, author.normalized_key),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def _apply_tags(conn: sqlite3.Connection, item_id: int, names: list[str]) -> None:
    for name in names:
        name = name.strip()
        if not name:
            continue
        conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (name,))
        tag_id = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()["id"]
        conn.execute(
            "INSERT OR IGNORE INTO item_tags(item_id, tag_id) VALUES (?, ?)",
            (item_id, tag_id),
        )


def _apply_collections(
    conn: sqlite3.Connection,
    item_id: int,
    collections: list[tuple[int, str, int | None]],
) -> None:
    """Upsert collections by name. Parent hierarchy is best-effort: if the
    parent is also in ``collections`` we link to it; otherwise the collection
    lands at the top level and the user can re-parent from the UI.

    A proper two-pass would resolve the full Zotero tree in one go, but the
    common case (leaf collections on items) works with this simpler pass."""
    col_ids_seen: dict[int, int] = {}  # zotero_id → grimoire_id
    for zot_id, name, zot_parent in collections:
        name = (name or "").strip()
        if not name:
            continue
        row = conn.execute(
            "SELECT id FROM collections WHERE name = ? AND parent_id IS NULL", (name,)
        ).fetchone()
        if row:
            grim_id = int(row["id"])
        else:
            parent_grim = col_ids_seen.get(zot_parent) if zot_parent else None
            cur = conn.execute(
                "INSERT INTO collections(name, parent_id) VALUES (?, ?)",
                (name, parent_grim),
            )
            grim_id = int(cur.lastrowid)  # type: ignore[arg-type]
        col_ids_seen[zot_id] = grim_id
        conn.execute(
            "INSERT OR IGNORE INTO item_collections(item_id, collection_id) "
            "VALUES (?, ?)",
            (item_id, grim_id),
        )


def _stamp_zotero_id(conn: sqlite3.Connection, item_id: int, zotero_item_id: int) -> None:
    """Merge ``{"zotero_item_id": N}`` into an existing item's metadata_json
    so re-running the migration skips it on the idempotency check."""
    row = conn.execute(
        "SELECT metadata_json FROM items WHERE id = ?", (item_id,)
    ).fetchone()
    existing = json.loads(row["metadata_json"]) if row and row["metadata_json"] else {}
    existing["zotero_item_id"] = zotero_item_id
    conn.execute(
        "UPDATE items SET metadata_json = ? WHERE id = ?", (json.dumps(existing), item_id)
    )


def _record(
    conn: sqlite3.Connection,
    source_path: str,
    content_hash: str | None,
    result: str,
    item_id: int,
) -> None:
    conn.execute(
        "INSERT INTO ingest_log(source_path, content_hash, result, item_id) "
        "VALUES (?, ?, ?, ?)",
        (source_path, content_hash, result, item_id),
    )
