"""Unit tests for each tier of grimoire.dedup (plan §3)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from tests.support.stub_embedder import StubEmbedder

from grimoire import dedup
from grimoire.embed.base import serialize_float32
from grimoire.models import Author, Metadata


def _insert(
    conn: sqlite3.Connection,
    *,
    title: str = "A paper",
    doi: str | None = None,
    arxiv_id: str | None = None,
    isbn: str | None = None,
    content_hash: str | None = None,
    authors: list[tuple[str, str | None]] | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO items(item_type, title, doi, arxiv_id, isbn, content_hash)
           VALUES ('paper', ?, ?, ?, ?, ?)""",
        (title, doi, arxiv_id, isbn, content_hash),
    )
    item_id = int(cur.lastrowid)  # type: ignore[arg-type]
    for pos, (last, first) in enumerate(authors or []):
        cur = conn.execute(
            """INSERT OR IGNORE INTO authors(family_name, given_name, normalized_key)
               VALUES (?, ?, ?)""",
            (last, first, f"{last.lower()},{(first or '')[:1].lower()}"),
        )
        author_id = (
            cur.lastrowid
            or conn.execute(
                "SELECT id FROM authors WHERE normalized_key=?",
                (f"{last.lower()},{(first or '')[:1].lower()}",),
            ).fetchone()["id"]
        )
        conn.execute(
            "INSERT INTO item_authors(item_id, author_id, position, role) VALUES (?, ?, ?, 'author')",
            (item_id, author_id, pos),
        )
    return item_id


# ---------- tier 1 --------------------------------------------------------


class TestTier1:
    def test_doi_match_merges(self, tmp_db: sqlite3.Connection) -> None:
        existing = _insert(tmp_db, doi="10.1234/x")
        d = dedup.decide(tmp_db, Metadata(title="new", doi="10.1234/x"), content_hash="h1")
        assert d.outcome == "merge"
        assert d.target_id == existing
        assert d.reason == "doi_match"

    def test_arxiv_match_merges(self, tmp_db: sqlite3.Connection) -> None:
        existing = _insert(tmp_db, arxiv_id="2401.00001")
        d = dedup.decide(tmp_db, Metadata(title="x", arxiv_id="2401.00001"), content_hash="h2")
        assert d.outcome == "merge" and d.target_id == existing and d.reason == "arxiv_id_match"

    def test_isbn_match_merges(self, tmp_db: sqlite3.Connection) -> None:
        existing = _insert(tmp_db, isbn="9780131101630")
        d = dedup.decide(tmp_db, Metadata(title="x", isbn="9780131101630"), content_hash="h3")
        assert d.outcome == "merge" and d.target_id == existing and d.reason == "isbn_match"

    def test_hash_match_merges(self, tmp_db: sqlite3.Connection) -> None:
        existing = _insert(tmp_db, content_hash="deadbeef")
        d = dedup.decide(tmp_db, Metadata(title="x"), content_hash="deadbeef")
        assert d.outcome == "merge" and d.target_id == existing and d.reason == "hash_match"

    def test_no_tier1_falls_through(self, tmp_db: sqlite3.Connection) -> None:
        _insert(tmp_db, doi="10.1234/a")
        d = dedup.decide(
            tmp_db, Metadata(title="different paper", doi="10.9999/b"), content_hash="h"
        )
        assert d.outcome == "insert"

    def test_exclude_self_match(self, tmp_db: sqlite3.Connection) -> None:
        """dedup-scan must not match an item against itself."""
        x = _insert(tmp_db, doi="10.1/z")
        d = dedup.decide(
            tmp_db, Metadata(title="x", doi="10.1/z"), content_hash="h", exclude_item_id=x
        )
        assert d.outcome == "insert"


# ---------- tier 2 --------------------------------------------------------


class TestTier2Erratum:
    def test_erratum_links_to_original(self, tmp_db: sqlite3.Connection) -> None:
        orig = _insert(tmp_db, title="Boron dilution transients in PWR reactors")
        d = dedup.decide(
            tmp_db,
            Metadata(title="Erratum to: Boron dilution transients in PWR reactors"),
            content_hash="h",
        )
        assert d.outcome == "link" and d.target_id == orig and d.relation == "erratum_for"

    def test_corrigendum_variant(self, tmp_db: sqlite3.Connection) -> None:
        orig = _insert(tmp_db, title="Fuel rod thermal modeling with CFD")
        d = dedup.decide(
            tmp_db,
            Metadata(title="Corrigendum: Fuel rod thermal modeling with CFD"),
            content_hash="h",
        )
        assert d.outcome == "link" and d.target_id == orig

    def test_correction_variant(self, tmp_db: sqlite3.Connection) -> None:
        orig = _insert(tmp_db, title="Light water reactor stability analysis")
        d = dedup.decide(
            tmp_db,
            Metadata(title="Correction to: Light water reactor stability analysis"),
            content_hash="h",
        )
        assert d.outcome == "link" and d.target_id == orig

    def test_too_short_suffix_ignored(self, tmp_db: sqlite3.Connection) -> None:
        _insert(tmp_db, title="Short")
        d = dedup.decide(tmp_db, Metadata(title="Erratum: foo"), content_hash="h")
        assert d.outcome == "insert"

    def test_no_matching_original(self, tmp_db: sqlite3.Connection) -> None:
        d = dedup.decide(
            tmp_db,
            Metadata(title="Erratum: a paper nobody has ever published before"),
            content_hash="h",
        )
        assert d.outcome == "insert"


# ---------- tier 3 --------------------------------------------------------


class TestTier3ArxivLinked:
    def test_preprint_links_to_published(self, tmp_db: sqlite3.Connection) -> None:
        published = _insert(tmp_db, doi="10.1016/j.published.2024")
        candidate = Metadata(
            title="X",
            arxiv_id="2401.00001",
            raw={"linked_doi": "10.1016/j.published.2024"},
        )
        d = dedup.decide(tmp_db, candidate, content_hash="h")
        assert d.outcome == "link" and d.target_id == published
        assert d.relation == "preprint_of"

    def test_no_linked_doi_key(self, tmp_db: sqlite3.Connection) -> None:
        candidate = Metadata(title="X", arxiv_id="2401.00001", raw={})
        d = dedup.decide(tmp_db, candidate, content_hash="h")
        assert d.outcome == "insert"

    def test_linked_doi_not_in_library(self, tmp_db: sqlite3.Connection) -> None:
        candidate = Metadata(
            title="X", arxiv_id="2401.00001", raw={"linked_doi": "10.9/nonexistent"}
        )
        d = dedup.decide(tmp_db, candidate, content_hash="h")
        assert d.outcome == "insert"


# ---------- tier 4 --------------------------------------------------------


def _insert_with_embedding(
    conn: sqlite3.Connection,
    title: str,
    vec: np.ndarray,
    authors: list[tuple[str, str | None]] | None = None,
) -> int:
    item_id = _insert(conn, title=title, authors=authors)
    conn.execute(
        "INSERT INTO item_embeddings(item_id, embedding) VALUES (?, ?)",
        (item_id, serialize_float32(vec)),
    )
    return item_id


class TestTier4Semantic:
    def test_high_sim_with_author_overlap_merges(self, tmp_db: sqlite3.Connection) -> None:
        """Cosine ≥ 0.97 and ≥ 1 shared author → MERGE."""
        target_vec = np.zeros(768, dtype=np.float32)
        target_vec[0] = 1.0
        near_vec = target_vec.copy()
        near_vec[1] = 0.05  # cos ≈ 0.9988

        existing = _insert_with_embedding(
            tmp_db, "boron dilution", target_vec, authors=[("Smith", "J")]
        )

        cand = Metadata(
            title="boron dilution (preprint)",
            authors=[Author(family_name="Smith", given_name="J")],
        )
        emb = StubEmbedder(dim=768, fixed={f"boron dilution (preprint) [SEP] {''}": near_vec})
        d = dedup.decide(tmp_db, cand, content_hash="h", item_embedder=emb)
        assert d.outcome == "merge"
        assert d.target_id == existing

    def test_high_sim_no_author_overlap_links_related(self, tmp_db: sqlite3.Connection) -> None:
        """Cosine ≥ 0.95, no author overlap → LINK(related) per plan §3."""
        base = np.zeros(768, dtype=np.float32)
        base[0] = 1.0
        near = base.copy()
        near[1] = 0.2  # cos ≈ 0.98

        existing = _insert_with_embedding(tmp_db, "deep X", base, authors=[("Smith", "J")])

        cand = Metadata(
            title="different title same topic",
            authors=[Author(family_name="Entirely", given_name="Different")],
        )
        emb = StubEmbedder(dim=768, fixed={f"different title same topic [SEP] {''}": near})
        d = dedup.decide(tmp_db, cand, content_hash="h", item_embedder=emb)
        assert d.outcome == "link"
        assert d.target_id == existing
        assert d.relation == "related"

    def test_ambiguous_band_without_judge_defers(self, tmp_db: sqlite3.Connection) -> None:
        """0.90 ≤ sim < 0.97 with author overlap and NO judge → fall through to insert."""
        base = np.zeros(768, dtype=np.float32)
        base[0] = 1.0
        mid = base.copy()
        mid[1] = 0.4  # cos ≈ 0.92

        _insert_with_embedding(tmp_db, "thing", base, authors=[("Smith", "J")])

        cand = Metadata(title="similar", authors=[Author(family_name="Smith", given_name="J")])
        emb = StubEmbedder(dim=768, fixed={f"similar [SEP] {''}": mid})
        d = dedup.decide(tmp_db, cand, content_hash="h", item_embedder=emb, llm_judge=None)
        assert d.outcome == "insert"

    def test_ambiguous_band_with_judge_same_merges(self, tmp_db: sqlite3.Connection) -> None:
        base = np.zeros(768, dtype=np.float32)
        base[0] = 1.0
        mid = base.copy()
        mid[1] = 0.4

        target = _insert_with_embedding(tmp_db, "thing", base, authors=[("Smith", "J")])

        cand = Metadata(
            title="same thing different wording",
            authors=[Author(family_name="Smith", given_name="J")],
        )
        emb = StubEmbedder(dim=768, fixed={f"same thing different wording [SEP] {''}": mid})
        d = dedup.decide(
            tmp_db,
            cand,
            content_hash="h",
            item_embedder=emb,
            llm_judge=lambda a, b: "same",
        )
        assert d.outcome == "merge"
        assert d.target_id == target
        assert d.reason == "llm_judge_same"

    def test_ambiguous_band_judge_related(self, tmp_db: sqlite3.Connection) -> None:
        base = np.zeros(768, dtype=np.float32)
        base[0] = 1.0
        mid = base.copy()
        mid[1] = 0.4
        target = _insert_with_embedding(tmp_db, "thing", base, authors=[("Smith", "J")])
        cand = Metadata(title="similar", authors=[Author(family_name="Smith", given_name="J")])
        emb = StubEmbedder(dim=768, fixed={f"similar [SEP] {''}": mid})
        d = dedup.decide(
            tmp_db,
            cand,
            content_hash="h",
            item_embedder=emb,
            llm_judge=lambda a, b: "related",
        )
        assert d.outcome == "link"
        assert d.target_id == target
        assert d.relation == "related"

    def test_ambiguous_band_judge_different_falls_through(self, tmp_db: sqlite3.Connection) -> None:
        base = np.zeros(768, dtype=np.float32)
        base[0] = 1.0
        mid = base.copy()
        mid[1] = 0.4
        _insert_with_embedding(tmp_db, "thing", base, authors=[("Smith", "J")])
        cand = Metadata(title="similar", authors=[Author(family_name="Smith", given_name="J")])
        emb = StubEmbedder(dim=768, fixed={f"similar [SEP] {''}": mid})
        d = dedup.decide(
            tmp_db,
            cand,
            content_hash="h",
            item_embedder=emb,
            llm_judge=lambda a, b: "different",
        )
        assert d.outcome == "insert"

    def test_non_duplicate_pair_respected(self, tmp_db: sqlite3.Connection) -> None:
        """User-asserted non_duplicate_pairs must block a tier-4 merge for
        re-scans (exclude_item_id is set)."""
        base = np.zeros(768, dtype=np.float32)
        base[0] = 1.0
        identical = base.copy()
        target = _insert_with_embedding(tmp_db, "p1", base, authors=[("Smith", "J")])
        candidate_id = _insert_with_embedding(tmp_db, "p2", identical, authors=[("Smith", "J")])
        lo, hi = sorted((target, candidate_id))
        tmp_db.execute("INSERT INTO non_duplicate_pairs(a_id, b_id) VALUES (?, ?)", (lo, hi))

        cand = Metadata(title="p2", authors=[Author(family_name="Smith", given_name="J")])
        emb = StubEmbedder(dim=768, fixed={f"p2 [SEP] {''}": identical})
        d = dedup.decide(
            tmp_db,
            cand,
            content_hash="h",
            item_embedder=emb,
            exclude_item_id=candidate_id,
        )
        assert d.outcome == "insert"


# ---------- apply_merge / apply_link -------------------------------------


class TestApply:
    def test_apply_merge_fills_missing(self, tmp_db: sqlite3.Connection) -> None:
        target = _insert(tmp_db, title="X", doi="10.1/x")
        cand = Metadata(
            title="X",
            abstract="an abstract that was missing before",
            doi="10.1/x",
            venue="Journal",
            publication_year=2024,
            authors=[Author(family_name="Alice", given_name=None)],
        )
        dedup.apply_merge(tmp_db, target, cand)
        row = tmp_db.execute(
            "SELECT abstract, venue, publication_year FROM items WHERE id=?", (target,)
        ).fetchone()
        assert row["abstract"] == "an abstract that was missing before"
        assert row["venue"] == "Journal"
        assert row["publication_year"] == 2024
        n_authors = tmp_db.execute(
            "SELECT COUNT(*) FROM item_authors WHERE item_id=?", (target,)
        ).fetchone()[0]
        assert n_authors == 1

    def test_apply_merge_doesnt_overwrite_existing(self, tmp_db: sqlite3.Connection) -> None:
        target = _insert(tmp_db, title="Original")
        tmp_db.execute("UPDATE items SET abstract=? WHERE id=?", ("original abstract", target))
        dedup.apply_merge(tmp_db, target, Metadata(title="X", abstract="a different abstract"))
        row = tmp_db.execute("SELECT abstract FROM items WHERE id=?", (target,)).fetchone()
        assert row["abstract"] == "original abstract"

    def test_apply_link_creates_symmetric_pair(self, tmp_db: sqlite3.Connection) -> None:
        a = _insert(tmp_db, title="preprint")
        b = _insert(tmp_db, title="published")
        dedup.apply_link(tmp_db, a, b, "preprint_of", 1.0)
        rels = tmp_db.execute(
            "SELECT subject_id, relation, object_id FROM item_relations ORDER BY subject_id"
        ).fetchall()
        assert len(rels) == 2
        # Forward: a preprint_of b; Inverse: b published_as a
        assert (rels[0]["subject_id"], rels[0]["relation"], rels[0]["object_id"]) == (
            a,
            "preprint_of",
            b,
        )
        assert (rels[1]["subject_id"], rels[1]["relation"], rels[1]["object_id"]) == (
            b,
            "published_as",
            a,
        )

    def test_apply_link_related_is_self_symmetric(self, tmp_db: sqlite3.Connection) -> None:
        a = _insert(tmp_db, title="p1")
        b = _insert(tmp_db, title="p2")
        dedup.apply_link(tmp_db, a, b, "related", 0.92)
        rels = tmp_db.execute("SELECT relation FROM item_relations").fetchall()
        assert [r["relation"] for r in rels] == ["related", "related"]

    def test_unknown_relation_rejected(self, tmp_db: sqlite3.Connection) -> None:
        a = _insert(tmp_db, title="p1")
        b = _insert(tmp_db, title="p2")
        with pytest.raises(ValueError):
            dedup.apply_link(tmp_db, a, b, "not_a_relation", 1.0)


# ---------- re-ingest idempotency (plan §6 Phase 3 oracle 3) --------------


def test_property_reingest_same_candidate_is_noop(tmp_db: sqlite3.Connection) -> None:
    """For any item x, dedup(ingest(x), ingest(x)) == no-op."""
    cand = Metadata(title="A nice paper", doi="10.1/nice", authors=[Author(family_name="Smith")])
    first = dedup.decide(tmp_db, cand, content_hash="hhh")
    assert first.outcome == "insert"

    # Simulate that insert happened.
    cur = tmp_db.execute(
        "INSERT INTO items(item_type, title, doi, content_hash) VALUES ('paper', ?, ?, ?)",
        (cand.title, cand.doi, "hhh"),
    )
    item_id = int(cur.lastrowid)  # type: ignore[arg-type]

    second = dedup.decide(tmp_db, cand, content_hash="hhh")
    assert second.outcome == "merge"
    assert second.target_id == item_id
    # Either hash_match (matched by content hash first) or doi_match is acceptable.
    assert second.reason in {"hash_match", "doi_match"}

    third = dedup.decide(tmp_db, cand, content_hash="hhh", exclude_item_id=item_id)
    assert third.outcome == "insert"  # exclude_item_id simulates "what if the item wasn't there"
