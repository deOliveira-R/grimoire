"""Ingestion pipeline: identifier extraction → resolvers → deterministic dedup → insert.

Phase 1 only implements the deterministic (tier 1) layer of the dedup algorithm
from plan §3: DOI / arXiv / ISBN / content-hash exact matches. Semantic dedup
(embeddings + LLM judge) arrives in Phase 3.

The pipeline is single-writer by design — the plan's §8 callout about
sqlite-vec's serial-write constraint applies here too."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from grimoire.config import settings
from grimoire.extract import pdf as pdf_extract
from grimoire.identify import identify
from grimoire.models import (
    Author,
    IngestResult,
    Metadata,
    prefer_more_authoritative,
)
from grimoire.resolve import arxiv_api, crossref, llm_fallback, openlibrary
from grimoire.storage.cas import CAS

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".epub"}


# ---------- public API -----------------------------------------------------


def ingest_file(conn: sqlite3.Connection, path: Path) -> IngestResult:
    """Ingest a single file. Idempotent by content hash and by resolved DOI/arXiv/ISBN."""
    path = path.resolve()
    if not path.is_file():
        return _fail(conn, path, None, "not a file")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return _fail(conn, path, None, f"unsupported suffix {path.suffix}")

    cas = CAS(settings.files_root)
    content_hash = CAS.hash_file(path)

    # Tier 1a: content-hash match. Exact byte-identical re-ingest → no-op.
    existing = conn.execute("SELECT id FROM items WHERE content_hash=?", (content_hash,)).fetchone()
    if existing:
        return _record(conn, path, content_hash, "skipped", existing["id"], "hash_match")

    # Resolve metadata.
    try:
        metadata = _resolve_metadata(path)
    except Exception as exc:
        log.exception("resolution failed for %s", path)
        return _fail(conn, path, content_hash, f"resolve error: {exc}")

    # Tier 1b-1d: deterministic identifier matches against existing items.
    matched = _match_by_identifier(conn, metadata)
    if matched is not None:
        return _record(conn, path, content_hash, "skipped", matched, "identifier_match")

    # Insert.
    _, _ = cas.store_file(path)
    item_id = _insert_item(conn, metadata, content_hash)
    _upsert_authors(conn, item_id, metadata.authors)
    return _record(conn, path, content_hash, "inserted", item_id, metadata.source)


def ingest_path(conn: sqlite3.Connection, root: Path, recursive: bool = True) -> list[IngestResult]:
    """Ingest a single file or walk a directory."""
    root = root.resolve()
    if root.is_file():
        return [ingest_file(conn, root)]
    return [ingest_file(conn, p) for p in _walk(root, recursive)]


# ---------- internals ------------------------------------------------------


def _walk(root: Path, recursive: bool) -> Iterable[Path]:
    it = root.rglob("*") if recursive else root.iterdir()
    for p in sorted(it):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p


def _resolve_metadata(path: Path) -> Metadata:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        first_page = pdf_extract.extract_first_page(path)
    else:
        # EPUBs don't have a page concept; take the text body and truncate so
        # regex extraction stays fast.
        from grimoire.extract import epub as epub_extract

        first_page = epub_extract.extract_text(path)[:8000]

    ids = identify(first_page)
    candidates: list[Metadata] = []

    for doi in ids.dois:
        md = crossref.resolve(doi)
        if md is not None:
            candidates.append(md)
            break  # One hit is enough; further DOIs on the page rarely refer to the same paper.

    for arxiv_id in ids.arxiv_ids:
        md = arxiv_api.resolve(arxiv_id)
        if md is not None:
            candidates.append(md)
            break

    for isbn in ids.isbns:
        md = openlibrary.resolve(isbn)
        if md is not None:
            candidates.append(md)
            break

    if not candidates:
        llm_md = llm_fallback.resolve(first_page)
        if llm_md is not None:
            candidates.append(llm_md)

    if candidates:
        return prefer_more_authoritative(candidates)

    # Nothing worked. Mark for manual review (oracle check 3).
    return Metadata(
        title=path.stem,
        source="manual_required",
        confidence=0.0,
        item_type=_guess_type(path),
    )


def _guess_type(path: Path) -> str:
    return "book" if path.suffix.lower() == ".epub" else "paper"


def _match_by_identifier(conn: sqlite3.Connection, md: Metadata) -> int | None:
    if md.doi:
        row = conn.execute("SELECT id FROM items WHERE doi=?", (md.doi,)).fetchone()
        if row:
            return int(row["id"])
    if md.arxiv_id:
        row = conn.execute("SELECT id FROM items WHERE arxiv_id=?", (md.arxiv_id,)).fetchone()
        if row:
            return int(row["id"])
    if md.isbn:
        row = conn.execute("SELECT id FROM items WHERE isbn=?", (md.isbn,)).fetchone()
        if row:
            return int(row["id"])
    return None


def _insert_item(conn: sqlite3.Connection, md: Metadata, content_hash: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO items(
            item_type, title, abstract, publication_year,
            doi, arxiv_id, isbn,
            venue, volume, issue, pages, series, series_number, edition, language,
            content_hash, file_path,
            metadata_source, metadata_confidence, metadata_json
        ) VALUES (?,?,?,?, ?,?,?, ?,?,?,?,?,?,?,?, ?,?, ?,?,?)
        """,
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
            _relative_cas_path(content_hash),
            md.source,
            md.confidence,
            json.dumps(md.raw) if md.raw is not None else None,
        ),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def _relative_cas_path(content_hash: str) -> str:
    return f"{content_hash[:2]}/{content_hash[2:4]}/{content_hash}"


def _upsert_authors(conn: sqlite3.Connection, item_id: int, authors: list[Author]) -> None:
    for position, author in enumerate(authors):
        if not author.family_name:
            continue
        author_id = _upsert_author(conn, author)
        conn.execute(
            """INSERT OR IGNORE INTO item_authors(item_id, author_id, position, role)
               VALUES (?,?,?, 'author')""",
            (item_id, author_id, position),
        )


def _upsert_author(conn: sqlite3.Connection, author: Author) -> int:
    if author.orcid:
        row = conn.execute("SELECT id FROM authors WHERE orcid=?", (author.orcid,)).fetchone()
        if row:
            return int(row["id"])

    # No orcid match: look up by normalized_key; accept name collisions
    # (plan §8 — disambiguation beyond last-name + first-initial is v2).
    row = conn.execute(
        "SELECT id FROM authors WHERE normalized_key=? AND orcid IS ?",
        (author.normalized_key, author.orcid),
    ).fetchone()
    if row:
        return int(row["id"])

    cur = conn.execute(
        """INSERT INTO authors(family_name, given_name, orcid, normalized_key)
           VALUES (?,?,?,?)""",
        (author.family_name, author.given_name, author.orcid, author.normalized_key),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def _record(
    conn: sqlite3.Connection,
    path: Path,
    content_hash: str | None,
    outcome: str,
    item_id: int | None,
    reason: str,
) -> IngestResult:
    conn.execute(
        """INSERT INTO ingest_log(source_path, content_hash, result, item_id)
           VALUES (?,?,?,?)""",
        (str(path), content_hash, outcome, item_id),
    )
    return IngestResult(
        outcome=outcome,  # type: ignore[arg-type]
        item_id=item_id,
        source_path=str(path),
        content_hash=content_hash,
        reason=reason,
    )


def _fail(
    conn: sqlite3.Connection,
    path: Path,
    content_hash: str | None,
    reason: str,
) -> IngestResult:
    conn.execute(
        """INSERT INTO ingest_log(source_path, content_hash, result)
           VALUES (?,?, 'failed')""",
        (str(path), content_hash),
    )
    log.warning("ingest failed for %s: %s", path, reason)
    return IngestResult(
        outcome="failed",
        item_id=None,
        source_path=str(path),
        content_hash=content_hash,
        reason=reason,
    )
