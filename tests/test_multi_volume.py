"""Multi-volume works: part_of / contains_part + series-parent auto-creation.

See plan §6 Phase 6. These tests lock in the behavior we'll lean on when the
full Phase 6 chapter-splitting work lands."""

from __future__ import annotations

import sqlite3

import pytest

from grimoire import dedup, ingest
from grimoire.models import Author, Metadata
from grimoire.resolve import crossref

# ---------- schema migration ----------------------------------------------


def test_part_of_relation_accepted_by_schema(tmp_db: sqlite3.Connection) -> None:
    """The 002 migration has added part_of/contains_part to the CHECK set."""
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('book', 'Vol I')")
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('book', 'TAOCP Set')")
    tmp_db.execute(
        "INSERT INTO item_relations(subject_id, relation, object_id) VALUES (1, 'part_of', 2)"
    )
    tmp_db.execute(
        "INSERT INTO item_relations(subject_id, relation, object_id) VALUES (2, 'contains_part', 1)"
    )
    rows = tmp_db.execute("SELECT relation FROM item_relations ORDER BY subject_id").fetchall()
    assert [r["relation"] for r in rows] == ["part_of", "contains_part"]


def test_invalid_relation_still_rejected(tmp_db: sqlite3.Connection) -> None:
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('book', 'X')")
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('book', 'Y')")
    with pytest.raises(sqlite3.IntegrityError):
        tmp_db.execute(
            "INSERT INTO item_relations(subject_id, relation, object_id) VALUES (1, 'not_a_rel', 2)"
        )


def test_apply_link_handles_part_of_symmetry(tmp_db: sqlite3.Connection) -> None:
    """dedup.apply_link must insert the inverse (contains_part) when given part_of."""
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('book', 'Vol 1')")
    tmp_db.execute("INSERT INTO items(item_type, title) VALUES ('book', 'Set')")
    dedup.apply_link(tmp_db, 1, 2, "part_of", 1.0)
    rels = tmp_db.execute(
        "SELECT subject_id, relation, object_id FROM item_relations ORDER BY subject_id"
    ).fetchall()
    assert [(r["subject_id"], r["relation"], r["object_id"]) for r in rels] == [
        (1, "part_of", 2),
        (2, "contains_part", 1),
    ]


# ---------- ingest-time parent auto-creation ------------------------------


@pytest.fixture
def stub_crossref_with_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub resolver that marks the item as a book with a series."""

    def fake(doi: str) -> Metadata | None:
        mapping = {
            "10.1234/taocp.vol1": (1997, "Volume 1: Fundamental Algorithms", "1"),
            "10.1234/taocp.vol2": (1997, "Volume 2: Seminumerical Algorithms", "2"),
            "10.1234/taocp.vol3": (1998, "Volume 3: Sorting and Searching", "3"),
        }
        if doi not in mapping:
            return None
        year, title, number = mapping[doi]
        return Metadata(
            title=title,
            publication_year=year,
            doi=doi,
            series="The Art of Computer Programming",
            series_number=number,
            item_type="book",
            authors=[Author(family_name="Knuth", given_name="Donald")],
            source="crossref",
            confidence=1.0,
        )

    monkeypatch.setattr(crossref, "resolve", fake)


def _make_volume_pdf(tmp_path, name: str, body: str):
    import pymupdf

    path = tmp_path / name
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), body)
    doc.save(str(path))
    doc.close()
    return path


def test_first_volume_creates_parent_set(
    tmp_db: sqlite3.Connection, tmp_path, stub_crossref_with_series: None
) -> None:
    pdf = _make_volume_pdf(tmp_path, "v1.pdf", "https://doi.org/10.1234/taocp.vol1")
    r = ingest.ingest_file(tmp_db, pdf)
    assert r.outcome == "inserted"

    # The parent set item exists, marked as derived
    parents = tmp_db.execute(
        "SELECT id, title, metadata_source FROM items WHERE metadata_source='derived'"
    ).fetchall()
    assert len(parents) == 1
    assert parents[0]["title"] == "The Art of Computer Programming"

    # Vol 1 is linked part_of the set; inverse contains_part also present
    rels = tmp_db.execute(
        "SELECT subject_id, relation, object_id FROM item_relations ORDER BY subject_id"
    ).fetchall()
    subj_rels = {(r["subject_id"], r["relation"]) for r in rels}
    assert (r.item_id, "part_of") in subj_rels


def test_second_volume_reuses_existing_parent(
    tmp_db: sqlite3.Connection, tmp_path, stub_crossref_with_series: None
) -> None:
    pdf1 = _make_volume_pdf(tmp_path, "v1.pdf", "https://doi.org/10.1234/taocp.vol1")
    pdf2 = _make_volume_pdf(tmp_path, "v2.pdf", "https://doi.org/10.1234/taocp.vol2")
    ingest.ingest_file(tmp_db, pdf1)
    ingest.ingest_file(tmp_db, pdf2)

    # Still exactly one derived parent
    parents = tmp_db.execute(
        "SELECT COUNT(*) FROM items WHERE metadata_source='derived'"
    ).fetchone()[0]
    assert parents == 1

    # Find ids dynamically — order depends on when the parent gets auto-created.
    parent_id = tmp_db.execute("SELECT id FROM items WHERE metadata_source='derived'").fetchone()[
        "id"
    ]
    volume_ids = {
        row["id"]
        for row in tmp_db.execute(
            "SELECT id FROM items WHERE metadata_source='crossref' AND item_type='book'"
        )
    }
    assert len(volume_ids) == 2

    # Both volumes point to the same parent
    parts = tmp_db.execute(
        "SELECT subject_id FROM item_relations WHERE relation='part_of' AND object_id=?",
        (parent_id,),
    ).fetchall()
    assert {r["subject_id"] for r in parts} == volume_ids

    # Set has contains_part to both volumes
    contains = tmp_db.execute(
        "SELECT object_id FROM item_relations WHERE relation='contains_part' AND subject_id=?",
        (parent_id,),
    ).fetchall()
    assert {r["object_id"] for r in contains} == volume_ids


def test_phantom_parent_not_embedded_by_index_all(
    tmp_db: sqlite3.Connection, tmp_path, stub_crossref_with_series: None
) -> None:
    """Derived parents have no abstract and aren't search targets; the backfill
    indexer should skip them so their title-only embedding doesn't pollute the
    vec0 space."""
    from tests.support.stub_embedder import StubEmbedder

    from grimoire.index import index_all

    pdf = _make_volume_pdf(tmp_path, "v1.pdf", "https://doi.org/10.1234/taocp.vol1")
    ingest.ingest_file(tmp_db, pdf)

    results = index_all(
        tmp_db, item_embedder=StubEmbedder(dim=768), chunk_embedder=StubEmbedder(dim=1024)
    )
    indexed_ids = {r.item_id for r in results}

    parent_id = tmp_db.execute("SELECT id FROM items WHERE metadata_source='derived'").fetchone()[
        "id"
    ]
    assert parent_id not in indexed_ids, "derived parent must be skipped"


def test_conservation_invariant_holds_with_phantoms(
    tmp_db: sqlite3.Connection, tmp_path, stub_crossref_with_series: None
) -> None:
    """Plan §7 invariant 1 survives synthetic parent creation because we log an
    'inserted' row for each phantom in ingest_log."""
    pdf = _make_volume_pdf(tmp_path, "v1.pdf", "https://doi.org/10.1234/taocp.vol1")
    ingest.ingest_file(tmp_db, pdf)

    items = tmp_db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    merges = tmp_db.execute("SELECT COUNT(*) FROM merge_history").fetchone()[0]
    new_item_events = tmp_db.execute(
        "SELECT COUNT(*) FROM ingest_log WHERE result IN ('inserted', 'linked')"
    ).fetchone()[0]
    assert items + merges == new_item_events


def test_paper_with_series_does_not_get_parent(tmp_db: sqlite3.Connection) -> None:
    """Conference proceedings often set 'series' on journal articles; we don't
    want a phantom set-parent for every conference."""
    md = Metadata(
        title="Some paper",
        doi="10.1/paper",
        series="Lecture Notes in Computer Science",
        item_type="paper",
        source="crossref",
        confidence=1.0,
    )
    from grimoire.ingest import _link_series_parent

    cur = tmp_db.execute(
        "INSERT INTO items(item_type, title, series, metadata_source) "
        "VALUES ('paper', ?, ?, 'crossref')",
        (md.title, md.series),
    )
    _link_series_parent(tmp_db, int(cur.lastrowid), md)

    parents = tmp_db.execute(
        "SELECT COUNT(*) FROM items WHERE metadata_source='derived'"
    ).fetchone()[0]
    assert parents == 0


# ---------- MCP structural kind -------------------------------------------


def test_mcp_list_related_structural_kind(
    tmp_db: sqlite3.Connection, tmp_path, stub_crossref_with_series: None
) -> None:
    from grimoire.mcp import tools as mcp_tools

    pdf1 = _make_volume_pdf(tmp_path, "v1.pdf", "https://doi.org/10.1234/taocp.vol1")
    pdf2 = _make_volume_pdf(tmp_path, "v2.pdf", "https://doi.org/10.1234/taocp.vol2")
    ingest.ingest_file(tmp_db, pdf1)
    ingest.ingest_file(tmp_db, pdf2)

    parent_id = tmp_db.execute("SELECT id FROM items WHERE metadata_source='derived'").fetchone()[
        "id"
    ]

    volume_ids = {
        row["id"] for row in tmp_db.execute("SELECT id FROM items WHERE metadata_source='crossref'")
    }

    # From the set: structural kind lists both volumes
    rel = mcp_tools.list_related(tmp_db, parent_id, kind="structural")
    assert {r.item_id for r in rel} == volume_ids
    assert {r.relation for r in rel} == {"contains_part"}

    # From a volume: structural kind lists the parent
    any_volume = next(iter(volume_ids))
    rel = mcp_tools.list_related(tmp_db, any_volume, kind="structural")
    assert len(rel) == 1
    assert rel[0].item_id == parent_id
    assert rel[0].relation == "part_of"
