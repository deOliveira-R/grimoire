"""Tier-4 edition detection (plan §6 Phase 6b).

When the candidate and nearest neighbor share authors and are close in
embedding space BUT declare different ``edition`` fields, we emit an
edition-of link instead of merging. Direction reflects which one is
newer: edition number first, then publication year, then a default."""

from __future__ import annotations

import sqlite3

import numpy as np

from tests.support.stub_embedder import StubEmbedder

from grimoire import dedup
from grimoire.embed.base import serialize_float32
from grimoire.models import Author, Metadata


def _insert(
    conn: sqlite3.Connection,
    *,
    title: str,
    vec: np.ndarray,
    edition: str | None = None,
    year: int | None = None,
    authors: list[tuple[str, str | None]] | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO items(item_type, title, edition, publication_year) "
        "VALUES ('book', ?, ?, ?)",
        (title, edition, year),
    )
    item_id = int(cur.lastrowid)
    for pos, (last, first) in enumerate(authors or []):
        c = conn.execute(
            "INSERT OR IGNORE INTO authors(family_name, given_name, normalized_key) "
            "VALUES (?, ?, ?)",
            (last, first, f"{last.lower()},{(first or '')[:1].lower()}"),
        )
        aid = (
            c.lastrowid
            or conn.execute(
                "SELECT id FROM authors WHERE normalized_key=?",
                (f"{last.lower()},{(first or '')[:1].lower()}",),
            ).fetchone()["id"]
        )
        conn.execute(
            "INSERT INTO item_authors(item_id, author_id, position, role) "
            "VALUES (?, ?, ?, 'author')",
            (item_id, aid, pos),
        )
    conn.execute(
        "INSERT INTO item_embeddings(item_id, embedding) VALUES (?, ?)",
        (item_id, serialize_float32(vec)),
    )
    return item_id


def _close_pair(noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit-normalized 768-d vectors with cosine ~ 1 - noise²/2.

    noise=0.05 → sim ≈ 0.999; noise=0.2 → sim ≈ 0.98; noise=0.4 → sim ≈ 0.92.
    """
    base = np.zeros(768, dtype=np.float32)
    base[0] = 1.0
    other = base.copy()
    other[1] = noise
    return base, other


def test_different_edition_numbers_link_later_edition_of(
    tmp_db: sqlite3.Connection,
) -> None:
    v1, v2 = _close_pair(noise=0.05)  # sim ≈ 0.999
    first = _insert(
        tmp_db,
        title="Reactor Physics",
        vec=v1,
        edition="1",
        year=2010,
        authors=[("Smith", "A")],
    )

    cand = Metadata(
        title="Reactor Physics",
        edition="3",
        publication_year=2022,
        authors=[Author(family_name="Smith", given_name="A")],
        item_type="book",
    )
    emb = StubEmbedder(dim=768, fixed={"Reactor Physics [SEP] ": v2})
    d = dedup.decide(tmp_db, cand, content_hash="h", item_embedder=emb)

    assert d.outcome == "link"
    assert d.target_id == first
    assert d.relation == "later_edition_of"
    assert d.reason.startswith("edition_sim_")


def test_candidate_older_edition_links_earlier_edition_of(
    tmp_db: sqlite3.Connection,
) -> None:
    v1, v2 = _close_pair(noise=0.05)
    newer = _insert(
        tmp_db,
        title="Reactor Physics",
        vec=v1,
        edition="3rd",
        year=2022,
        authors=[("Smith", "A")],
    )
    cand = Metadata(
        title="Reactor Physics",
        edition="1st",
        publication_year=2010,
        authors=[Author(family_name="Smith", given_name="A")],
        item_type="book",
    )
    emb = StubEmbedder(dim=768, fixed={"Reactor Physics [SEP] ": v2})
    d = dedup.decide(tmp_db, cand, content_hash="h", item_embedder=emb)

    assert d.outcome == "link"
    assert d.target_id == newer
    assert d.relation == "earlier_edition_of"


def test_only_one_side_has_edition_field_still_links(
    tmp_db: sqlite3.Connection,
) -> None:
    v1, v2 = _close_pair(noise=0.05)
    first = _insert(
        tmp_db,
        title="Thermal analysis",
        vec=v1,
        edition=None,
        year=2005,
        authors=[("Doe", "B")],
    )
    cand = Metadata(
        title="Thermal analysis",
        edition="2nd",
        publication_year=2020,
        authors=[Author(family_name="Doe", given_name="B")],
        item_type="book",
    )
    emb = StubEmbedder(dim=768, fixed={"Thermal analysis [SEP] ": v2})
    d = dedup.decide(tmp_db, cand, content_hash="h", item_embedder=emb)
    assert d.outcome == "link"
    assert d.target_id == first
    assert d.relation == "later_edition_of"


def test_same_edition_string_does_not_link(tmp_db: sqlite3.Connection) -> None:
    v1, v2 = _close_pair(noise=0.05)
    _insert(
        tmp_db,
        title="Fluid dynamics",
        vec=v1,
        edition="2",
        authors=[("Zhao", "C")],
    )
    cand = Metadata(
        title="Fluid dynamics",
        edition="2nd",  # same edition number, different phrasing
        authors=[Author(family_name="Zhao", given_name="C")],
        item_type="book",
    )
    emb = StubEmbedder(dim=768, fixed={"Fluid dynamics [SEP] ": v2})
    d = dedup.decide(tmp_db, cand, content_hash="h", item_embedder=emb)
    # With matching normalized editions, the edition path doesn't fire;
    # sim≈0.999 and author overlap → falls through to the merge path.
    assert d.outcome == "merge"


def test_no_edition_info_on_either_side_does_not_fire(
    tmp_db: sqlite3.Connection,
) -> None:
    v1, v2 = _close_pair(noise=0.05)
    _insert(tmp_db, title="Nothing", vec=v1, edition=None, authors=[("X", "Y")])
    cand = Metadata(
        title="Nothing",
        edition=None,
        authors=[Author(family_name="X", given_name="Y")],
        item_type="book",
    )
    emb = StubEmbedder(dim=768, fixed={"Nothing [SEP] ": v2})
    d = dedup.decide(tmp_db, cand, content_hash="h", item_embedder=emb)
    # Falls through to the usual merge threshold.
    assert d.outcome == "merge"


def test_low_similarity_does_not_fire_edition_path(
    tmp_db: sqlite3.Connection,
) -> None:
    v1, v2 = _close_pair(noise=0.8)  # sim ≈ 0.68 — below the 0.85 threshold
    _insert(
        tmp_db,
        title="Mostly unrelated",
        vec=v1,
        edition="1",
        authors=[("Alpha", "A")],
    )
    cand = Metadata(
        title="Mostly unrelated",
        edition="2",
        authors=[Author(family_name="Alpha", given_name="A")],
        item_type="book",
    )
    emb = StubEmbedder(dim=768, fixed={"Mostly unrelated [SEP] ": v2})
    d = dedup.decide(tmp_db, cand, content_hash="h", item_embedder=emb)
    assert d.outcome == "insert"


def test_edition_path_requires_author_overlap(tmp_db: sqlite3.Connection) -> None:
    # High sim, no author overlap → existing "related" path fires; no edition link.
    v1, v2 = _close_pair(noise=0.2)
    _insert(tmp_db, title="Topic X", vec=v1, edition="1", authors=[("Alpha", "A")])
    cand = Metadata(
        title="Topic X",
        edition="2",
        authors=[Author(family_name="Omega", given_name="O")],
        item_type="book",
    )
    emb = StubEmbedder(dim=768, fixed={"Topic X [SEP] ": v2})
    d = dedup.decide(tmp_db, cand, content_hash="h", item_embedder=emb)
    assert d.outcome == "link"
    assert d.relation == "related"  # not edition


# ---------- apply_link creates the symmetric inverse ------------------------


def test_apply_link_creates_symmetric_edition_pair(tmp_db: sqlite3.Connection) -> None:
    a = tmp_db.execute(
        "INSERT INTO items(item_type, title) VALUES ('book', 'A') RETURNING id"
    ).fetchone()["id"]
    b = tmp_db.execute(
        "INSERT INTO items(item_type, title) VALUES ('book', 'B') RETURNING id"
    ).fetchone()["id"]
    dedup.apply_link(tmp_db, a, b, "later_edition_of", 0.9)
    rels = tmp_db.execute(
        "SELECT subject_id, relation, object_id FROM item_relations ORDER BY subject_id"
    ).fetchall()
    assert len(rels) == 2
    relations = {r["relation"] for r in rels}
    assert relations == {"later_edition_of", "earlier_edition_of"}
