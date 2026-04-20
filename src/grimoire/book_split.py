"""Turn a freshly-ingested book into book + N chapter items (plan §6 Phase 6a).

Called from ``ingest._act`` after a book is inserted. Reads the CAS blob,
tries ``extract.book_structure.detect`` for chapter boundaries, and — if
detection succeeds — creates one ``item_type='chapter'`` per entry with a
``chapter_of`` relation back to the book.

Chapters do NOT get their own ``content_hash`` / ``file_path``. The indexer
re-materializes their text from the parent book's CAS blob using the page
range / spine href stashed in ``metadata_json``. Downloads happen at book
granularity; the web UI surfaces a chapter's parent-book via the
``chapter_of`` relation."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from grimoire import dedup
from grimoire.config import settings
from grimoire.extract.book_structure import ChapterSpec, detect
from grimoire.storage.cas import CAS

log = logging.getLogger(__name__)


def split_book(
    conn: sqlite3.Connection, book_item_id: int, content_hash: str
) -> list[int]:
    """Detect chapters and create one item per chapter.

    Returns the chapter item_ids (empty if no structure detected)."""
    cas = CAS(settings.files_root)
    path = cas.path_for_hash(content_hash)
    if not path.exists():
        log.warning("book_split: CAS blob missing for %s", content_hash)
        return []

    # Chapter extraction is best-effort: a malformed TOC shouldn't fail
    # the enclosing ingest.
    try:
        specs = detect(path)
    except Exception as exc:
        log.warning("book_split: structure detection failed for item %d: %s", book_item_id, exc)
        return []
    if not specs:
        return []

    book_title = _book_title(conn, book_item_id) or f"item_{book_item_id}"
    chapter_ids: list[int] = []
    for spec in specs:
        chapter_id = _insert_chapter(conn, book_item_id, spec)
        _log_chapter(conn, book_title, spec, chapter_id)
        dedup.apply_link(conn, chapter_id, book_item_id, "chapter_of", 1.0)
        chapter_ids.append(chapter_id)
    return chapter_ids


# ---------- internals --------------------------------------------------------


def _book_title(conn: sqlite3.Connection, item_id: int) -> str | None:
    row = conn.execute("SELECT title FROM items WHERE id=?", (item_id,)).fetchone()
    return row["title"] if row else None


def _insert_chapter(
    conn: sqlite3.Connection, parent_id: int, spec: ChapterSpec
) -> int:
    """Insert a chapter row. ``metadata_json`` carries the page range /
    spine href the indexer needs to reconstruct the chapter's text."""
    meta: dict[str, int | str] = {
        "parent_book_id": parent_id,
        "chapter_index": spec.index,
    }
    if spec.start_page is not None and spec.end_page is not None:
        meta["start_page"] = spec.start_page
        meta["end_page"] = spec.end_page
    if spec.spine_href is not None:
        meta["spine_href"] = spec.spine_href

    cur = conn.execute(
        """INSERT INTO items(item_type, title, metadata_source, metadata_confidence,
                             metadata_json)
           VALUES ('chapter', ?, 'book-split', 0.95, ?)""",
        (spec.title or f"Chapter {spec.index + 1}", json.dumps(meta)),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def _log_chapter(
    conn: sqlite3.Connection,
    book_title: str,
    spec: ChapterSpec,
    chapter_id: int,
) -> None:
    """Each chapter gets its own ``ingest_log`` entry so the conservation
    invariant (plan §7 #1: ``count(items) + count(merge_history) ==
    count(ingest_log)``) keeps holding after splitting."""
    source = f"<book-split: {book_title}#{spec.index}>"
    conn.execute(
        "INSERT INTO ingest_log(source_path, content_hash, result, item_id) "
        "VALUES (?, NULL, 'inserted', ?)",
        (source, chapter_id),
    )


# ---------- indexer support --------------------------------------------------


def chapter_pages(conn: sqlite3.Connection, chapter_item_id: int) -> list[tuple[int, str]]:
    """Reconstruct a chapter's text from the parent book's CAS blob.

    Returns ``[(page_number, text), ...]`` like ``index._extract_pages``.
    Returns an empty list for chapters whose parent book or CAS blob is
    missing, or whose metadata_json has been lost."""
    row = conn.execute(
        "SELECT metadata_json FROM items WHERE id = ? AND item_type = 'chapter'",
        (chapter_item_id,),
    ).fetchone()
    if not row or not row["metadata_json"]:
        return []
    meta = json.loads(row["metadata_json"])
    parent_id = meta.get("parent_book_id")
    if parent_id is None:
        return []

    parent_row = conn.execute(
        "SELECT content_hash FROM items WHERE id = ?", (parent_id,)
    ).fetchone()
    if not parent_row or not parent_row["content_hash"]:
        return []

    cas = CAS(settings.files_root)
    path = cas.path_for_hash(parent_row["content_hash"])
    if not path.exists():
        return []

    if "start_page" in meta:
        from grimoire.extract.book_structure import pdf_chapter_text

        return pdf_chapter_text(path, int(meta["start_page"]), int(meta["end_page"]))

    if "spine_href" in meta:
        from grimoire.extract.book_structure import epub_chapter_text

        text = epub_chapter_text(path, meta["spine_href"])
        return [(meta.get("chapter_index", 0) + 1, text)] if text.strip() else []

    return []
