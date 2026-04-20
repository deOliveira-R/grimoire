"""Ingestion pipeline: identifier extraction → resolvers → tiered dedup → insert.

Implements plan §3 in full: tier-1 deterministic match (DOI / arXiv / ISBN /
hash) → tier-2 erratum regex → tier-3 arXiv preprint↔published linking → tier-4
semantic (SPECTER2 + author-overlap + optional LLM judge).

Tier-4 needs a loaded embedder, so it's opt-in via the ``item_embedder``
parameter. For single-file CLI ingestion we default to deterministic-only so
the user doesn't pay the model-load cost on every invocation. Batch migration
(``grimoire index``, ``grimoire dedup-scan``) pass the embedder in.

The pipeline is single-writer by design — plan §8 callout about sqlite-vec's
serial-write constraint applies here too."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from grimoire import dedup
from grimoire.config import settings
from grimoire.dedup import DedupDecision, JudgeFn
from grimoire.embed.base import Embedder
from grimoire.extract import grobid
from grimoire.extract import pdf as pdf_extract
from grimoire.identify import identify
from grimoire.models import (
    Author,
    IngestResult,
    Metadata,
    merge_metadata_layered,
)
from grimoire.resolve import arxiv_api, crossref, llm_fallback, openlibrary
from grimoire.storage.cas import CAS

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".epub"}


# ---------- public API -----------------------------------------------------


def ingest_file(
    conn: sqlite3.Connection,
    path: Path,
    *,
    item_embedder: Embedder | None = None,
    llm_judge: JudgeFn | None = None,
) -> IngestResult:
    """Ingest a single file through the tiered dedup algorithm (plan §3).

    If ``item_embedder`` is provided, tier-4 semantic dedup runs. Without it,
    only deterministic tiers 1-3 run — which is the desired default for the
    single-file CLI path."""
    path = path.resolve()
    if not path.is_file():
        return _fail(conn, path, None, "not a file")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return _fail(conn, path, None, f"unsupported suffix {path.suffix}")

    cas = CAS(settings.files_root)
    content_hash = CAS.hash_file(path)

    # Short-circuit: byte-identical re-ingest doesn't need to hit Crossref/GROBID
    # again. This is tier-1 hash_match semantically — we still log as 'merged'
    # to match plan §3, but the DB is already consistent and apply_merge would
    # no-op.
    existing = conn.execute("SELECT id FROM items WHERE content_hash=?", (content_hash,)).fetchone()
    if existing:
        return _record(conn, path, content_hash, "merged", int(existing["id"]), "hash_match")

    try:
        metadata = _resolve_metadata(path)
    except Exception as exc:
        log.exception("resolution failed for %s", path)
        return _fail(conn, path, content_hash, f"resolve error: {exc}")

    decision = dedup.decide(
        conn,
        metadata,
        content_hash,
        item_embedder=item_embedder,
        llm_judge=llm_judge,
    )

    return _act(conn, decision, path, content_hash, metadata, cas, item_embedder)


def ingest_path(
    conn: sqlite3.Connection,
    root: Path,
    recursive: bool = True,
    *,
    item_embedder: Embedder | None = None,
    llm_judge: JudgeFn | None = None,
) -> list[IngestResult]:
    """Ingest a single file or walk a directory."""
    root = root.resolve()
    files = [root] if root.is_file() else list(_walk(root, recursive))
    return [ingest_file(conn, p, item_embedder=item_embedder, llm_judge=llm_judge) for p in files]


def _act(
    conn: sqlite3.Connection,
    decision: DedupDecision,
    path: Path,
    content_hash: str,
    metadata: Metadata,
    cas: CAS,
    item_embedder: Embedder | None,
) -> IngestResult:
    if decision.outcome == "insert":
        cas.store_file(path)
        item_id = _insert_item(conn, metadata, content_hash)
        _upsert_authors(conn, item_id, metadata.authors)
        _link_series_parent(conn, item_id, metadata)
        if item_embedder is not None:
            _embed_and_store(conn, item_id, metadata, item_embedder)
        return _record(conn, path, content_hash, "inserted", item_id, decision.reason)

    if decision.outcome == "merge":
        target = decision.target_id
        assert target is not None
        # Store the file even though no new item is created — the bytes are
        # kept in CAS so we don't lose the source. CAS dedups by hash so if the
        # content is identical, this is a no-op on disk.
        cas.store_file(path)
        dedup.apply_merge(conn, target, metadata)
        # Post-merge: the target may have just gained a series field from the
        # candidate, so the parent-link step can newly apply.
        _link_series_parent(conn, target, metadata)
        return _record(conn, path, content_hash, "merged", target, decision.reason)

    if decision.outcome == "link":
        target = decision.target_id
        relation = decision.relation
        assert target is not None and relation is not None
        cas.store_file(path)
        item_id = _insert_item(conn, metadata, content_hash)
        _upsert_authors(conn, item_id, metadata.authors)
        _link_series_parent(conn, item_id, metadata)
        if item_embedder is not None:
            _embed_and_store(conn, item_id, metadata, item_embedder)
        dedup.apply_link(conn, item_id, target, relation, decision.confidence)
        return _record(conn, path, content_hash, "linked", item_id, decision.reason)

    # skip
    return _record(conn, path, content_hash, "skipped", decision.target_id, decision.reason)


def _link_series_parent(conn: sqlite3.Connection, item_id: int, metadata: Metadata) -> None:
    """If this item declares a series, find or auto-create a synthetic `book`
    parent for that series and link via `part_of` (plan §6 Phase 6, multi-
    volume works).

    The parent item has no file/DOI — it's a logical grouping. metadata_source
    is 'derived' so indexers and dedup scans can skip it.
    """
    series = (metadata.series or "").strip()
    if not series:
        return
    # Only volume-style items get a parent. Papers occasionally carry a series
    # field (book series of proceedings) but we don't want a synthetic parent
    # per conference.
    if metadata.item_type not in {"book", "chapter", "report"}:
        return

    row = conn.execute(
        """SELECT id FROM items
           WHERE metadata_source = 'derived' AND title = ? AND item_type = 'book'""",
        (series,),
    ).fetchone()
    if row is not None:
        parent_id = int(row["id"])
    else:
        cur = conn.execute(
            """INSERT INTO items(item_type, title, metadata_source, metadata_confidence)
               VALUES ('book', ?, 'derived', 0.5)""",
            (series,),
        )
        parent_id = int(cur.lastrowid)  # type: ignore[arg-type]
        # Log the derived row so the conservation invariant (plan §7 #1) holds
        # without having to special-case metadata_source in the test.
        conn.execute(
            """INSERT INTO ingest_log(source_path, content_hash, result, item_id)
               VALUES (?, NULL, 'inserted', ?)""",
            (f"<auto-series: {series}>", parent_id),
        )

    if parent_id == item_id:
        return  # shouldn't happen, but guard anyway
    dedup.apply_link(conn, item_id, parent_id, "part_of", 1.0)


def _embed_and_store(
    conn: sqlite3.Connection, item_id: int, metadata: Metadata, embedder: Embedder
) -> None:
    """Embed the new item immediately so later ingestions in the same batch
    can find it via tier-4 semantic search."""
    from grimoire.embed.base import l2_normalize, serialize_float32
    from grimoire.embed.specter2 import format_item_text

    text = format_item_text(metadata.title or "", metadata.abstract)
    vec = l2_normalize(embedder.encode([text]))[0]
    conn.execute(
        "INSERT OR REPLACE INTO item_embeddings(item_id, embedding) VALUES (?, ?)",
        (item_id, serialize_float32(vec)),
    )


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

    candidates: list[Metadata] = []

    # GROBID first on PDFs — gives us title+abstract directly from the header,
    # bypassing Crossref's ~99% abstract-miss rate (Phase 2 oracle finding).
    if suffix == ".pdf" and settings.grobid_url:
        grobid_md = grobid.extract_header(path)
        if grobid_md is not None:
            candidates.append(grobid_md)

    # Collect identifier candidates from GROBID's output *and* the regex pass,
    # so the downstream resolvers get the union.
    ids = identify(first_page)
    dois = list(ids.dois)
    if candidates and candidates[0].doi and candidates[0].doi not in dois:
        dois.insert(0, candidates[0].doi)
    arxiv_ids = list(ids.arxiv_ids)
    isbns = list(ids.isbns)

    for doi in dois:
        md = crossref.resolve(doi)
        if md is not None:
            candidates.append(md)
            break  # One hit is enough; further DOIs on the page rarely refer to the same paper.

    for arxiv_id in arxiv_ids:
        md = arxiv_api.resolve(arxiv_id)
        if md is not None:
            candidates.append(md)
            break

    for isbn in isbns:
        md = openlibrary.resolve(isbn)
        if md is not None:
            candidates.append(md)
            break

    # LLM fallback only when nothing else produced metadata.
    if not candidates:
        llm_md = llm_fallback.resolve(first_page)
        if llm_md is not None:
            candidates.append(llm_md)

    if candidates:
        return merge_metadata_layered(candidates)

    # Nothing worked. Mark for manual review (oracle check 3).
    return Metadata(
        title=path.stem,
        source="manual_required",
        confidence=0.0,
        item_type=_guess_type(path),
    )


def _guess_type(path: Path) -> str:
    return "book" if path.suffix.lower() == ".epub" else "paper"


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
    # Delegates to the dedup module's author helpers so ingest-time and
    # dedup-time author handling stay identical.
    dedup._union_authors(conn, item_id, authors)


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
