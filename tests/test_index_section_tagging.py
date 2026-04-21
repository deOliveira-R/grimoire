"""Section-aware chunking: when a ``grobid_tei`` artifact is available, each
chunk gets tagged with the classified section type. Items without a TEI
artifact fall back to per-page chunking with section=NULL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.support.stub_embedder import StubEmbedder

from grimoire import index as index_mod
from grimoire import ingest
from grimoire.models import Author, Metadata
from grimoire.resolve import crossref
from grimoire.storage import artifacts


_TEI_WITH_SECTIONS = b"""<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Paper with body</title></titleStmt>
      <sourceDesc><biblStruct><analytic>
        <author><persName><surname>Solo</surname></persName></author>
      </analytic></biblStruct></sourceDesc>
    </fileDesc>
  </teiHeader>
  <text><body>
    <div><head n="1">Introduction</head>
      <p>The problem domain begins here and spans at least a few words
      to survive the chunker's sentence packer.</p></div>
    <div><head n="2">Materials and Methods</head>
      <p>We used spectrographs and thermocouples. The calibration procedure
      is described below in detail.</p></div>
    <div><head n="3">Results</head>
      <p>Measured values fell within the expected range. We observed a clear
      trend above the baseline.</p></div>
  </body></text>
</TEI>
"""


def _seed(tmp_db: sqlite3.Connection, pdf: Path, monkeypatch: object) -> int:
    def fake_doi(doi: str) -> Metadata | None:
        return Metadata(
            title="Paper with body",
            abstract="abstract",
            publication_year=2024,
            doi=doi,
            authors=[Author(family_name="Solo")],
            source="crossref",
            confidence=1.0,
        )

    monkeypatch.setattr(crossref, "resolve", fake_doi)  # type: ignore[attr-defined]
    r = ingest.ingest_file(tmp_db, pdf)
    assert r.item_id is not None
    return r.item_id


def test_tei_backed_chunks_are_section_tagged(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, monkeypatch: object
) -> None:
    item_id = _seed(tmp_db, pdf_with_doi, monkeypatch)
    artifacts.store(tmp_db, item_id, "grobid_tei", _TEI_WITH_SECTIONS, source="test")

    index_mod.index_item(
        tmp_db,
        item_id,
        item_embedder=StubEmbedder(dim=768),
        chunk_embedder=StubEmbedder(dim=1024),
    )

    rows = tmp_db.execute(
        "SELECT section FROM chunks WHERE item_id=? ORDER BY chunk_index",
        (item_id,),
    ).fetchall()
    assert len(rows) >= 3
    sections = {r["section"] for r in rows}
    # Section-aware path ran: every chunk carries a non-NULL section tag.
    assert None not in sections
    # The three TEI sections map to the classifier's canonical labels.
    assert {"introduction", "methods", "results"}.issubset(sections)


def test_fallback_chunks_have_null_section(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, monkeypatch: object
) -> None:
    """No TEI artifact → per-page fallback → section is NULL everywhere."""
    item_id = _seed(tmp_db, pdf_with_doi, monkeypatch)

    index_mod.index_item(
        tmp_db,
        item_id,
        item_embedder=StubEmbedder(dim=768),
        chunk_embedder=StubEmbedder(dim=1024),
    )

    rows = tmp_db.execute(
        "SELECT section FROM chunks WHERE item_id=?",
        (item_id,),
    ).fetchall()
    assert rows  # fallback still produced chunks from the PDF body
    assert all(r["section"] is None for r in rows)
