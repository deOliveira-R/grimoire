"""Book-to-chapters split (plan §6 Phase 6a).

Covers both halves of the flow:
  * ``extract.book_structure.detect`` on synthetic PDFs (with/without TOC)
    and EPUBs (with spine + nav).
  * ``book_split.split_book`` end-to-end: chapter items land with the right
    relations, ingest_log rows hold the conservation invariant, and the
    indexer can re-materialize per-chapter text through
    ``book_split.chapter_pages``."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from grimoire import book_split
from grimoire.extract.book_structure import ChapterSpec, detect


# ---------- fixtures --------------------------------------------------------


def _pdf_with_toc(path: Path, chapters: list[tuple[str, int]], total_pages: int) -> Path:
    """Build a multi-page PDF and attach a top-level TOC.

    ``chapters`` is ``[(title, start_page), ...]`` (1-indexed).
    """
    import pymupdf

    doc = pymupdf.open()
    for p in range(total_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {p + 1} body text.")
    doc.set_toc([[1, title, start] for title, start in chapters])
    doc.save(str(path))
    doc.close()
    return path


def _pdf_without_toc(path: Path, total_pages: int) -> Path:
    import pymupdf

    doc = pymupdf.open()
    for p in range(total_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {p + 1}")
    doc.save(str(path))
    doc.close()
    return path


def _multi_chapter_epub(path: Path, titles: list[str]) -> Path:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("multi-1")
    book.set_title("Multi-chapter test")
    book.set_language("en")
    book.add_author("Test Author")

    chapters: list[epub.EpubHtml] = []
    for i, title in enumerate(titles):
        ch = epub.EpubHtml(title=title, file_name=f"ch{i}.xhtml", lang="en")
        ch.content = f"<h1>{title}</h1><p>Body of chapter {i}.</p>"
        book.add_item(ch)
        chapters.append(ch)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *chapters]

    epub.write_epub(str(path), book, {})
    return path


# ---------- extract.book_structure.detect -----------------------------------


def test_detect_pdf_with_toc(tmp_path: Path) -> None:
    pdf = _pdf_with_toc(
        tmp_path / "book.pdf",
        chapters=[("Intro", 1), ("Methods", 4), ("Results", 8)],
        total_pages=10,
    )
    specs = detect(pdf)
    assert specs is not None
    assert [s.title for s in specs] == ["Intro", "Methods", "Results"]
    assert [(s.start_page, s.end_page) for s in specs] == [(1, 3), (4, 7), (8, 10)]
    assert [s.index for s in specs] == [0, 1, 2]


def test_detect_pdf_without_toc_returns_none(tmp_path: Path) -> None:
    pdf = _pdf_without_toc(tmp_path / "no_toc.pdf", total_pages=5)
    assert detect(pdf) is None


def test_detect_pdf_with_single_chapter_returns_none(tmp_path: Path) -> None:
    pdf = _pdf_with_toc(
        tmp_path / "single.pdf", chapters=[("Whole book", 1)], total_pages=3
    )
    assert detect(pdf) is None


def test_detect_epub_spine(tmp_path: Path) -> None:
    ep = _multi_chapter_epub(tmp_path / "book.epub", ["Alpha", "Beta", "Gamma"])
    specs = detect(ep)
    assert specs is not None
    assert len(specs) == 3
    # Titles derived from the NCX / nav TOC
    assert [s.title for s in specs] == ["Alpha", "Beta", "Gamma"]
    assert all(s.spine_href for s in specs)
    assert all(s.start_page is None for s in specs)


def test_detect_unsupported_extension_returns_none(tmp_path: Path) -> None:
    plain = tmp_path / "note.txt"
    plain.write_text("not a book")
    assert detect(plain) is None


# ---------- book_split.split_book integration --------------------------------


def _seed_book(
    conn: sqlite3.Connection, content_hash: str, title: str = "Multi book"
) -> int:
    cur = conn.execute(
        """INSERT INTO items(item_type, title, content_hash, file_path,
                             metadata_source, metadata_confidence)
           VALUES ('book', ?, ?, ?, 'manual', 1.0)""",
        (title, content_hash, f"{content_hash[:2]}/{content_hash[2:4]}/{content_hash}"),
    )
    item_id = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO ingest_log(source_path, content_hash, result, item_id) "
        "VALUES (?, ?, 'inserted', ?)",
        (title, content_hash, item_id),
    )
    return item_id


def test_split_book_creates_chapter_items(
    tmp_db: sqlite3.Connection, tmp_data_root: Path, tmp_path: Path
) -> None:
    from grimoire.storage.cas import CAS

    pdf = _pdf_with_toc(
        tmp_path / "book.pdf",
        chapters=[("Intro", 1), ("Methods", 5), ("Results", 9)],
        total_pages=12,
    )
    cas = CAS(tmp_data_root / "files")
    h, _ = cas.store_file(pdf)
    book_id = _seed_book(tmp_db, h, "Reactor physics")

    ids = book_split.split_book(tmp_db, book_id, h)

    assert len(ids) == 3
    # Chapter items carry item_type='chapter' and metadata_source='book-split'
    for cid in ids:
        row = tmp_db.execute(
            "SELECT item_type, metadata_source FROM items WHERE id=?", (cid,)
        ).fetchone()
        assert row["item_type"] == "chapter"
        assert row["metadata_source"] == "book-split"

    # chapter_of + contains_chapter inverse links exist for each
    for cid in ids:
        chapter_of = tmp_db.execute(
            "SELECT 1 FROM item_relations WHERE subject_id=? AND relation='chapter_of' AND object_id=?",
            (cid, book_id),
        ).fetchone()
        assert chapter_of is not None
        inverse = tmp_db.execute(
            "SELECT 1 FROM item_relations WHERE subject_id=? AND relation='contains_chapter' AND object_id=?",
            (book_id, cid),
        ).fetchone()
        assert inverse is not None


def test_split_book_logs_conservation_invariant(
    tmp_db: sqlite3.Connection, tmp_data_root: Path, tmp_path: Path
) -> None:
    from grimoire.storage.cas import CAS

    pdf = _pdf_with_toc(
        tmp_path / "book.pdf",
        chapters=[("Ch1", 1), ("Ch2", 3), ("Ch3", 5)],
        total_pages=6,
    )
    cas = CAS(tmp_data_root / "files")
    h, _ = cas.store_file(pdf)
    book_id = _seed_book(tmp_db, h)

    items_before = tmp_db.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"]
    log_before = tmp_db.execute("SELECT COUNT(*) AS n FROM ingest_log").fetchone()["n"]

    book_split.split_book(tmp_db, book_id, h)

    items_after = tmp_db.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"]
    log_after = tmp_db.execute("SELECT COUNT(*) AS n FROM ingest_log").fetchone()["n"]

    # Every new chapter item gets its own ingest_log row (plan §7 invariant 1)
    assert items_after - items_before == 3
    assert log_after - log_before == 3


def test_split_book_no_structure_is_noop(
    tmp_db: sqlite3.Connection, tmp_data_root: Path, tmp_path: Path
) -> None:
    from grimoire.storage.cas import CAS

    pdf = _pdf_without_toc(tmp_path / "plain.pdf", total_pages=3)
    cas = CAS(tmp_data_root / "files")
    h, _ = cas.store_file(pdf)
    book_id = _seed_book(tmp_db, h)

    ids = book_split.split_book(tmp_db, book_id, h)
    assert ids == []
    # No chapter items were created
    assert (
        tmp_db.execute(
            "SELECT COUNT(*) AS n FROM items WHERE item_type='chapter'"
        ).fetchone()["n"]
        == 0
    )


def test_split_book_missing_cas_blob_is_safe(tmp_db: sqlite3.Connection) -> None:
    # Phantom hash, no file on disk — should log and no-op, not raise.
    book_id = _seed_book(tmp_db, "a" * 64)
    assert book_split.split_book(tmp_db, book_id, "a" * 64) == []


# ---------- chapter_pages --------------------------------------------------


def test_chapter_pages_reconstructs_pdf_text(
    tmp_db: sqlite3.Connection, tmp_data_root: Path, tmp_path: Path
) -> None:
    from grimoire.storage.cas import CAS

    pdf = _pdf_with_toc(
        tmp_path / "book.pdf",
        chapters=[("A", 1), ("B", 3), ("C", 5)],
        total_pages=6,
    )
    cas = CAS(tmp_data_root / "files")
    h, _ = cas.store_file(pdf)
    book_id = _seed_book(tmp_db, h)
    ids = book_split.split_book(tmp_db, book_id, h)

    # Chapter B spans pages 3–4 of the original book
    pages = book_split.chapter_pages(tmp_db, ids[1])
    assert [p for p, _ in pages] == [3, 4]
    joined = "\n".join(text for _, text in pages)
    assert "Page 3" in joined
    assert "Page 4" in joined
    assert "Page 5" not in joined


def test_chapter_pages_for_non_chapter_returns_empty(
    tmp_db: sqlite3.Connection,
) -> None:
    cur = tmp_db.execute(
        "INSERT INTO items(item_type, title, metadata_source, metadata_confidence) "
        "VALUES ('paper', 'Not a chapter', 'manual', 1.0)"
    )
    paper_id = int(cur.lastrowid)
    assert book_split.chapter_pages(tmp_db, paper_id) == []


# ---------- end-to-end via ingest pipeline -----------------------------------


def test_ingest_book_splits_chapters_end_to_end(
    tmp_db: sqlite3.Connection, tmp_data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ingest a book PDF through the real pipeline: detect, split, link."""
    from grimoire import ingest

    pdf = _pdf_with_toc(
        tmp_path / "whole_book.pdf",
        chapters=[("Part I", 1), ("Part II", 4), ("Part III", 7)],
        total_pages=9,
    )

    # Force the metadata resolver to produce a book (bypasses Crossref/GROBID).
    from grimoire.models import Metadata

    def _fake_resolve(path: Path) -> Metadata:
        return Metadata(title=path.stem, source="manual", confidence=1.0, item_type="book")

    monkeypatch.setattr(ingest, "_resolve_metadata", _fake_resolve)

    result = ingest.ingest_file(tmp_db, pdf)
    assert result.outcome == "inserted"
    assert result.item_id is not None

    chapters = tmp_db.execute(
        "SELECT id, title FROM items WHERE item_type='chapter' ORDER BY id"
    ).fetchall()
    assert len(chapters) == 3
    assert [c["title"] for c in chapters] == ["Part I", "Part II", "Part III"]
