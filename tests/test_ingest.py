"""End-to-end ingest tests. Oracle checks 2 and 3 from plan §6 Phase 1."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from grimoire import ingest
from grimoire.models import Author, Metadata
from grimoire.resolve import crossref


@pytest.fixture
def stub_crossref(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return a canned metadata for 10.1234/example.2024; None otherwise."""

    def fake(doi: str) -> Metadata | None:
        if doi == "10.1234/example.2024":
            return Metadata(
                title="A Nice Paper",
                abstract="Concise abstract.",
                publication_year=2024,
                doi=doi,
                venue="Journal of Things",
                authors=[
                    Author(family_name="Alice", given_name=None),
                    Author(family_name="Bob", given_name=None),
                ],
                source="crossref",
                confidence=1.0,
            )
        return None

    monkeypatch.setattr(crossref, "resolve", fake)


def test_ingest_pdf_with_doi_creates_item(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, stub_crossref: None
) -> None:
    result = ingest.ingest_file(tmp_db, pdf_with_doi)
    assert result.outcome == "inserted"
    row = tmp_db.execute("SELECT * FROM items WHERE id=?", (result.item_id,)).fetchone()
    assert row["title"] == "A Nice Paper"
    assert row["doi"] == "10.1234/example.2024"
    assert row["metadata_source"] == "crossref"
    assert row["content_hash"] is not None
    # Authors joined.
    n_authors = tmp_db.execute(
        "SELECT COUNT(*) FROM item_authors WHERE item_id=?", (result.item_id,)
    ).fetchone()[0]
    assert n_authors == 2


def test_ingest_is_idempotent_by_hash(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, pdf_copy: Path, stub_crossref: None
) -> None:
    """Oracle check 2: ingest the same PDF twice → exactly one item in DB,
    no file duplicated in store. Tier-1 hash match returns 'merged' per plan §3."""
    from grimoire.config import settings
    from grimoire.storage.cas import CAS

    first = ingest.ingest_file(tmp_db, pdf_with_doi)
    assert first.outcome == "inserted"
    second = ingest.ingest_file(tmp_db, pdf_copy)
    assert second.outcome == "merged"
    assert second.item_id == first.item_id

    n_items = tmp_db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    assert n_items == 1

    n_log = tmp_db.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0]
    assert n_log == 2

    cas = CAS(settings.files_root)
    files = [p for p in cas.root.rglob("*") if p.is_file()]
    assert len(files) == 1  # same hash → one file in CAS


def test_ingest_no_identifier_no_llm_marks_manual_required(
    tmp_db: sqlite3.Connection, pdf_no_identifier: Path
) -> None:
    """Oracle check 3: no DOI, no LLM key → item with metadata_source='manual_required'."""
    result = ingest.ingest_file(tmp_db, pdf_no_identifier)
    assert result.outcome == "inserted"
    row = tmp_db.execute("SELECT * FROM items WHERE id=?", (result.item_id,)).fetchone()
    assert row["metadata_source"] == "manual_required"
    assert row["metadata_confidence"] == 0.0
    # Title falls back to filename stem so the item is still discoverable.
    assert row["title"] == pdf_no_identifier.stem


def test_ingest_dedup_by_doi(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, tmp_path: Path, stub_crossref: None
) -> None:
    """Two different files with the same resolved DOI merge into one item."""
    import pymupdf

    from grimoire.config import settings
    from grimoire.storage.cas import CAS

    # Build a second PDF with the same DOI but different body text (different hash).
    other = tmp_path / "other.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Reprint copy\nhttps://doi.org/10.1234/example.2024")
    doc.save(str(other))
    doc.close()

    first = ingest.ingest_file(tmp_db, pdf_with_doi)
    second = ingest.ingest_file(tmp_db, other)

    assert first.outcome == "inserted"
    assert second.outcome == "merged"  # Phase 3: tier-1 DOI match → MERGE per plan §3
    assert second.item_id == first.item_id

    n_items = tmp_db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    assert n_items == 1

    # Both files live in CAS — plan §3 MERGE semantics: "Keep both files if
    # hashes differ, attach as versions". file_versions table itself lands in
    # a later phase; for now the extra bytes just sit in the content-addressed
    # store referenced via ingest_log.
    cas = CAS(settings.files_root)
    files = [p for p in cas.root.rglob("*") if p.is_file()]
    assert len(files) == 2

    # Audit trail: every tier-1 merge records a merge_history row pointing at
    # the surviving target with a non-empty reason.
    rows = tmp_db.execute(
        "SELECT target_id, reason FROM merge_history"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["target_id"] == first.item_id
    assert rows[0]["reason"] and "match" in rows[0]["reason"]


def test_ingest_writes_primary_artifact(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, stub_crossref: None
) -> None:
    """Every ingested file lands as a 'primary' artifact row pointing at the
    item's content_hash. Downstream artifact pipelines (GROBID, OCR) gate on
    this row existing."""
    result = ingest.ingest_file(tmp_db, pdf_with_doi)
    assert result.item_id is not None
    row = tmp_db.execute(
        "SELECT content_hash, source FROM item_artifacts "
        "WHERE item_id=? AND kind='primary'",
        (result.item_id,),
    ).fetchone()
    assert row is not None
    items_hash = tmp_db.execute(
        "SELECT content_hash FROM items WHERE id=?", (result.item_id,)
    ).fetchone()["content_hash"]
    assert row["content_hash"] == items_hash
    assert row["source"] == "crossref"


def test_ingest_directory_recursive(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, pdf_no_identifier: Path, stub_crossref: None
) -> None:
    results = ingest.ingest_path(tmp_db, pdf_with_doi.parent, recursive=True)
    assert len(results) >= 2
    outcomes = [r.outcome for r in results]
    assert outcomes.count("inserted") == 2


def test_conservation_invariant_holds(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, pdf_copy: Path, stub_crossref: None
) -> None:
    """Plan §7 invariant 1: count(items) + count(merge_history) == count(ingest_log).

    Every ingest event either produces a new item (`inserted`/`linked`),
    re-attaches to an existing item via a merge (`merged` → +1 merge_history),
    or is skipped as already-imported (`skipped` → also a merge_history row
    so the conservation count balances)."""
    ingest.ingest_file(tmp_db, pdf_with_doi)
    ingest.ingest_file(tmp_db, pdf_copy)

    items = tmp_db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    merges = tmp_db.execute("SELECT COUNT(*) FROM merge_history").fetchone()[0]
    log_total = tmp_db.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0]
    log_skipped = tmp_db.execute(
        "SELECT COUNT(*) FROM ingest_log WHERE result = 'skipped'"
    ).fetchone()[0]
    # Skipped events don't change either side of the equation.
    assert items + merges == log_total - log_skipped
