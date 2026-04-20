"""Unit tests for the MCP tool implementations. These hit the pure-impl
functions directly so tests don't need an MCP protocol round-trip."""

from __future__ import annotations

import sqlite3

import pytest

from grimoire.embed.base import l2_normalize, serialize_float32
from grimoire.mcp import tools
from grimoire.mcp.citation import to_bibtex


@pytest.fixture
def seeded_db(tmp_db: sqlite3.Connection) -> sqlite3.Connection:
    """Corpus with three items + chunks + a relation + a tag + a collection."""
    import numpy as np

    def add_item(
        item_type: str,
        title: str,
        abstract: str | None = None,
        year: int | None = None,
        doi: str | None = None,
        arxiv_id: str | None = None,
        venue: str | None = None,
    ) -> int:
        cur = tmp_db.execute(
            """INSERT INTO items(item_type, title, abstract, publication_year,
                                  doi, arxiv_id, venue, metadata_source, metadata_confidence)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (item_type, title, abstract, year, doi, arxiv_id, venue, "crossref", 1.0),
        )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def add_author(item_id: int, family: str, given: str | None = None, pos: int = 0) -> None:
        key = f"{family.lower()},{(given or '')[:1].lower()}"
        tmp_db.execute(
            "INSERT OR IGNORE INTO authors(family_name, given_name, normalized_key) VALUES (?,?,?)",
            (family, given, key),
        )
        aid = tmp_db.execute(
            "SELECT id FROM authors WHERE normalized_key=? AND orcid IS ?", (key, None)
        ).fetchone()["id"]
        tmp_db.execute(
            "INSERT INTO item_authors(item_id, author_id, position, role) VALUES (?,?,?, 'author')",
            (item_id, aid, pos),
        )

    p1 = add_item(
        "paper",
        "Boron dilution transients in PWR",
        abstract="A study of boron concentration during accident conditions.",
        year=2024,
        doi="10.1/boron",
        venue="Nucl. Eng. Design",
    )
    add_author(p1, "Smith", "Alice", 0)
    add_author(p1, "Doe", "Bob", 1)

    p2 = add_item(
        "preprint",
        "Boron dilution transients in PWR (preprint)",
        year=2023,
        arxiv_id="2301.11111",
    )
    add_author(p2, "Smith", "Alice", 0)

    book = add_item("book", "Nuclear reactor physics handbook", year=2020)
    add_author(book, "Editor", "Em", 0)

    # Chunks for p1 body text
    for idx, (page, text) in enumerate(
        [
            (1, "boron dilution creates reactivity excursions"),
            (2, "control rod insertion mitigates the event"),
            (3, "three-dimensional modeling required for accuracy"),
        ]
    ):
        tmp_db.execute(
            "INSERT INTO chunks(item_id, chunk_index, page, text) VALUES (?, ?, ?, ?)",
            (p1, idx, page, text),
        )

    # Item embedding for p1 so semantic search has at least one match
    rng = np.random.default_rng(42)
    vec = l2_normalize(rng.standard_normal((1, 768)).astype("float32"))[0]
    tmp_db.execute(
        "INSERT INTO item_embeddings(item_id, embedding) VALUES (?, ?)",
        (p1, serialize_float32(vec)),
    )

    # Relation: p2 preprint_of p1
    tmp_db.execute(
        "INSERT INTO item_relations(subject_id, relation, object_id, confidence) VALUES (?,?,?,?)",
        (p2, "preprint_of", p1, 1.0),
    )
    tmp_db.execute(
        "INSERT INTO item_relations(subject_id, relation, object_id, confidence) VALUES (?,?,?,?)",
        (p1, "published_as", p2, 1.0),
    )

    # Tag + collection
    tmp_db.execute("INSERT INTO tags(name) VALUES ('safety')")
    tmp_db.execute(
        "INSERT INTO item_tags(item_id, tag_id) SELECT ?, id FROM tags WHERE name='safety'",
        (p1,),
    )
    tmp_db.execute("INSERT INTO collections(name) VALUES ('Reactor safety')")
    tmp_db.execute(
        "INSERT INTO item_collections(item_id, collection_id) "
        "SELECT ?, id FROM collections WHERE name='Reactor safety'",
        (p1,),
    )

    return tmp_db


# ---------- get_item / search ---------------------------------------------


def test_get_item_returns_full_metadata(seeded_db: sqlite3.Connection) -> None:
    item = tools.get_item(seeded_db, 1)
    assert item is not None
    assert item.title.startswith("Boron dilution")
    assert item.abstract and "boron concentration" in item.abstract
    assert item.authors == ["Smith", "Doe"]
    assert item.doi == "10.1/boron"
    assert item.venue == "Nucl. Eng. Design"
    assert item.year == 2024
    assert "safety" in item.tags
    assert "Reactor safety" in item.collections


def test_get_item_missing(seeded_db: sqlite3.Connection) -> None:
    assert tools.get_item(seeded_db, 9999) is None


def test_search_keyword(seeded_db: sqlite3.Connection) -> None:
    hits = tools.search(seeded_db, "boron", mode="keyword", limit=5)
    # Both the paper and the preprint have "boron" in the title
    assert len(hits) >= 1
    assert any(h.item_id == 1 for h in hits)


def test_search_with_item_type_filter(seeded_db: sqlite3.Connection) -> None:
    hits = tools.search(seeded_db, "boron", mode="keyword", item_types=["preprint"], limit=5)
    assert all(h.item_type == "preprint" for h in hits)


def test_search_includes_snippet_when_chunks_match(seeded_db: sqlite3.Connection) -> None:
    # "boron" matches the item title (ranks item 1 in the result) and also
    # matches chunk 1's text → best-snippet hydration attaches page=1.
    hits = tools.search(seeded_db, "boron dilution", mode="keyword", limit=5)
    main = next((h for h in hits if h.item_id == 1), None)
    assert main is not None
    assert main.snippet is not None
    assert main.snippet.page == 1


# ---------- get_full_text -------------------------------------------------


def test_full_text_all_pages(seeded_db: sqlite3.Connection) -> None:
    text = tools.get_full_text(seeded_db, 1)
    assert "boron dilution" in text
    assert "control rod" in text
    assert "three-dimensional" in text


def test_full_text_single_page(seeded_db: sqlite3.Connection) -> None:
    text = tools.get_full_text(seeded_db, 1, page=2)
    assert "control rod" in text
    assert "boron" not in text  # page 1 content excluded


def test_full_text_no_chunks_returns_empty(seeded_db: sqlite3.Connection) -> None:
    assert tools.get_full_text(seeded_db, 2) == ""


# ---------- get_snippets --------------------------------------------------


def test_snippets_keyword_fallback(seeded_db: sqlite3.Connection) -> None:
    snips = tools.get_snippets(seeded_db, "reactivity", k=5)
    assert snips
    assert snips[0].page == 1
    assert "reactivity" in snips[0].text


def test_snippets_item_scoped(seeded_db: sqlite3.Connection) -> None:
    snips = tools.get_snippets(seeded_db, "control", item_id=1, k=5)
    assert all(s.item_id == 1 for s in snips)


def test_snippets_empty_query(seeded_db: sqlite3.Connection) -> None:
    assert tools.get_snippets(seeded_db, "  ", k=5) == []


# ---------- list_related / relation filtering -----------------------------


def test_list_related_all(seeded_db: sqlite3.Connection) -> None:
    rel = tools.list_related(seeded_db, 1)
    assert len(rel) == 1
    assert rel[0].item_id == 2
    assert rel[0].relation == "published_as"


def test_list_related_preprint_chain(seeded_db: sqlite3.Connection) -> None:
    # From the preprint: should find the published version
    rel = tools.list_related(seeded_db, 2, kind="preprint_chain")
    assert len(rel) == 1
    assert rel[0].relation == "preprint_of"
    assert rel[0].item_id == 1


def test_list_related_citations_empty(seeded_db: sqlite3.Connection) -> None:
    assert tools.list_related(seeded_db, 1, kind="citations") == []


# ---------- tags / collections --------------------------------------------


def test_list_tags(seeded_db: sqlite3.Connection) -> None:
    assert tools.list_tags(seeded_db) == ["safety"]


def test_find_by_tag(seeded_db: sqlite3.Connection) -> None:
    items = tools.find_by_tag(seeded_db, "safety")
    assert len(items) == 1
    assert items[0].item_id == 1


def test_find_by_unknown_tag(seeded_db: sqlite3.Connection) -> None:
    assert tools.find_by_tag(seeded_db, "nonexistent-tag") == []


def test_list_collections(seeded_db: sqlite3.Connection) -> None:
    cols = tools.list_collections(seeded_db)
    assert len(cols) == 1
    assert cols[0].name == "Reactor safety"
    assert cols[0].item_count == 1


# ---------- BibTeX --------------------------------------------------------


def test_bibtex_article(seeded_db: sqlite3.Connection) -> None:
    bib = to_bibtex(seeded_db, 1)
    assert bib is not None
    assert bib.startswith("@article{")
    assert "smith_2024" in bib  # key is firstauthor_year_firstword
    assert "doi = {10.1/boron}" in bib
    # Title is double-braced so BibTeX preserves case
    assert "{Boron dilution transients in PWR}" in bib
    assert "journal = {Nucl. Eng. Design}" in bib
    assert "author = {Smith and Doe}" in bib


def test_bibtex_book(seeded_db: sqlite3.Connection) -> None:
    bib = to_bibtex(seeded_db, 3)
    assert bib is not None
    assert bib.startswith("@book{")


def test_bibtex_missing_item(seeded_db: sqlite3.Connection) -> None:
    assert to_bibtex(seeded_db, 9999) is None


def test_bibtex_escapes_special_chars(seeded_db: sqlite3.Connection) -> None:
    seeded_db.execute(
        "INSERT INTO items(item_type, title, metadata_source) "
        "VALUES ('paper', 'Costs & effects: 50% of samples', 'manual')"
    )
    new_id = seeded_db.execute("SELECT last_insert_rowid()").fetchone()[0]
    bib = to_bibtex(seeded_db, new_id)
    assert bib is not None
    assert r"\&" in bib
    assert r"\%" in bib
