from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from grimoire import db as db_module


@pytest.fixture
def tmp_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point grimoire.config.settings at a tmp data root."""
    monkeypatch.setattr("grimoire.config.settings.data_root", tmp_path)
    monkeypatch.setattr("grimoire.config.settings.anthropic_api_key", None)
    return tmp_path


@pytest.fixture
def tmp_db(tmp_data_root: Path) -> sqlite3.Connection:
    """Fresh database with all migrations applied, isolated to tmp_path."""
    conn = db_module.connect(tmp_data_root / "db" / "library.db")
    db_module.apply_migrations(conn)
    return conn


def _make_pdf(path: Path, body: str) -> Path:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), body)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf_with_doi(tmp_path: Path) -> Path:
    return _make_pdf(
        tmp_path / "with_doi.pdf",
        "A Nice Paper\nAlice, Bob\nhttps://doi.org/10.1234/example.2024",
    )


@pytest.fixture
def pdf_no_identifier(tmp_path: Path) -> Path:
    return _make_pdf(
        tmp_path / "no_id.pdf",
        "Unknown Paper\nNo identifier whatsoever",
    )


@pytest.fixture
def pdf_copy(pdf_with_doi: Path, tmp_path: Path) -> Path:
    """Byte-identical copy at a different path."""
    dst = tmp_path / "subdir" / "duplicate.pdf"
    dst.parent.mkdir()
    dst.write_bytes(pdf_with_doi.read_bytes())
    return dst


@pytest.fixture
def sample_epub(tmp_path: Path) -> Path:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("test-1")
    book.set_title("Sample Book")
    book.set_language("en")
    book.add_author("Jane Doe")

    chapter = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
    chapter.content = "<h1>Chapter 1</h1><p>Hello world, this is a sample chapter.</p>"
    book.add_item(chapter)
    book.toc = (chapter,)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]

    path = tmp_path / "sample.epub"
    epub.write_epub(str(path), book, {})
    return path
