"""Backfill pipeline: embed items + chunks using StubEmbedder. Avoids model loads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from tests.support.stub_embedder import StubEmbedder

from grimoire import index as index_mod
from grimoire import ingest
from grimoire.models import Author, Metadata
from grimoire.resolve import crossref


def _seed(tmp_db: sqlite3.Connection, pdf: Path, monkeypatch: object) -> int:
    def fake_doi(doi: str) -> Metadata | None:
        if doi == "10.1234/example.2024":
            return Metadata(
                title="Paper with body",
                abstract="The original abstract.",
                publication_year=2024,
                doi=doi,
                authors=[Author(family_name="Solo")],
                source="crossref",
                confidence=1.0,
            )
        return None

    # monkeypatch is a pytest.MonkeyPatch fixture but we pass it in for reuse
    monkeypatch.setattr(crossref, "resolve", fake_doi)  # type: ignore[attr-defined]
    r = ingest.ingest_file(tmp_db, pdf)
    assert r.item_id is not None
    return r.item_id


def test_index_item_populates_embedding(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, monkeypatch: object
) -> None:
    item_id = _seed(tmp_db, pdf_with_doi, monkeypatch)
    item_embedder = StubEmbedder(dim=768)
    chunk_embedder = StubEmbedder(dim=1024)

    index_mod.index_item(
        tmp_db, item_id, item_embedder=item_embedder, chunk_embedder=chunk_embedder
    )

    count = tmp_db.execute(
        "SELECT COUNT(*) FROM item_embeddings WHERE item_id=?", (item_id,)
    ).fetchone()[0]
    assert count == 1

    # Pipeline writes chunks + chunk_embeddings for items whose file is in CAS.
    n_chunks = tmp_db.execute("SELECT COUNT(*) FROM chunks WHERE item_id=?", (item_id,)).fetchone()[
        0
    ]
    assert n_chunks >= 1
    n_chunk_embs = tmp_db.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
    assert n_chunk_embs == n_chunks


def test_reindex_is_idempotent(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, monkeypatch: object
) -> None:
    item_id = _seed(tmp_db, pdf_with_doi, monkeypatch)
    item_embedder = StubEmbedder(dim=768)
    chunk_embedder = StubEmbedder(dim=1024)

    index_mod.index_item(
        tmp_db, item_id, item_embedder=item_embedder, chunk_embedder=chunk_embedder
    )
    before_chunks = tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    before_embs = tmp_db.execute("SELECT COUNT(*) FROM item_embeddings").fetchone()[0]

    # Re-run with force=False: no change.
    index_mod.index_item(
        tmp_db, item_id, item_embedder=item_embedder, chunk_embedder=chunk_embedder
    )
    after_chunks = tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    after_embs = tmp_db.execute("SELECT COUNT(*) FROM item_embeddings").fetchone()[0]
    assert before_chunks == after_chunks
    assert before_embs == after_embs


def test_force_reindex_replaces(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, monkeypatch: object
) -> None:
    item_id = _seed(tmp_db, pdf_with_doi, monkeypatch)

    item_embedder_a = StubEmbedder(dim=768)
    chunk_embedder_a = StubEmbedder(dim=1024)
    index_mod.index_item(
        tmp_db, item_id, item_embedder=item_embedder_a, chunk_embedder=chunk_embedder_a
    )
    before = tmp_db.execute(
        "SELECT embedding FROM item_embeddings WHERE item_id=?", (item_id,)
    ).fetchone()[0]

    item_embedder_b = StubEmbedder(
        dim=768,
        fixed={"Paper with body [SEP] The original abstract.": np.ones(768, dtype=np.float32)},
    )
    index_mod.index_item(
        tmp_db,
        item_id,
        item_embedder=item_embedder_b,
        chunk_embedder=chunk_embedder_a,
        force=True,
    )
    after = tmp_db.execute(
        "SELECT embedding FROM item_embeddings WHERE item_id=?", (item_id,)
    ).fetchone()[0]
    assert before != after


def test_index_all_skips_already_indexed(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, monkeypatch: object
) -> None:
    item_id = _seed(tmp_db, pdf_with_doi, monkeypatch)
    item_embedder = StubEmbedder(dim=768)
    chunk_embedder = StubEmbedder(dim=1024)

    r1 = index_mod.index_all(tmp_db, item_embedder=item_embedder, chunk_embedder=chunk_embedder)
    assert any(r.item_id == item_id and r.status == "indexed" for r in r1)

    r2 = index_mod.index_all(tmp_db, item_embedder=item_embedder, chunk_embedder=chunk_embedder)
    assert all(r.status == "skipped" for r in r2)


def test_conservation_invariant_8_fts_counts_hold(
    tmp_db: sqlite3.Connection, pdf_with_doi: Path, monkeypatch: object
) -> None:
    """Plan §7 invariant 8: count(items)==count(items_fts); count(chunks)==count(chunks_fts)."""
    item_id = _seed(tmp_db, pdf_with_doi, monkeypatch)
    index_mod.index_item(
        tmp_db, item_id, item_embedder=StubEmbedder(dim=768), chunk_embedder=StubEmbedder(dim=1024)
    )
    items = tmp_db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    items_fts = tmp_db.execute("SELECT COUNT(*) FROM items_fts").fetchone()[0]
    chunks = tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    chunks_fts = tmp_db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert items == items_fts
    assert chunks == chunks_fts
