"""Hybrid facade: combines keyword + semantic over a small seeded DB using StubEmbedder.
Avoids any real model load, so it runs fast in CI."""

from __future__ import annotations

import sqlite3

import numpy as np
from tests.support.stub_embedder import StubEmbedder

from grimoire.embed.base import l2_normalize, serialize_float32
from grimoire.search import search_items


def _seed(conn: sqlite3.Connection, stub: StubEmbedder) -> list[int]:
    items = [
        ("Boron dilution transients in PWR", "study of boron concentration"),
        ("Vegetable recipes", "cooking with carrots"),
        ("Fuel rod thermal modeling", "rod heat transfer analysis"),
    ]
    ids = []
    for title, abstract in items:
        cur = conn.execute(
            "INSERT INTO items(item_type, title, abstract) VALUES ('paper', ?, ?)",
            (title, abstract),
        )
        item_id = int(cur.lastrowid)  # type: ignore[arg-type]
        emb = l2_normalize(stub.encode([f"{title} [SEP] {abstract}"]))
        conn.execute(
            "INSERT INTO item_embeddings(item_id, embedding) VALUES (?, ?)",
            (item_id, serialize_float32(emb[0])),
        )
        ids.append(item_id)
    return ids


def test_keyword_mode_matches_title(tmp_db: sqlite3.Connection) -> None:
    stub = StubEmbedder(dim=768)
    ids = _seed(tmp_db, stub)
    hits = search_items(tmp_db, "boron", mode="keyword", limit=5, item_embedder=stub)
    assert hits[0].item_id == ids[0]


def test_semantic_mode_finds_nearest(tmp_db: sqlite3.Connection) -> None:
    # Prime the stub with a known vector for the query, and place the target
    # item's embedding near it so semantic search surfaces it first.
    query_vec = np.zeros(768, dtype=np.float32)
    query_vec[0] = 1.0
    near_vec = query_vec.copy()
    near_vec[1] = 0.01

    stub = StubEmbedder(
        dim=768,
        fixed={
            "fuel rod thermal analysis": query_vec,
            "Fuel rod thermal modeling [SEP] rod heat transfer analysis": near_vec,
        },
    )
    ids = _seed(tmp_db, stub)

    hits = search_items(
        tmp_db, "fuel rod thermal analysis", mode="semantic", limit=5, item_embedder=stub
    )
    assert hits[0].item_id == ids[2]


def test_hybrid_mode_runs_end_to_end(tmp_db: sqlite3.Connection) -> None:
    stub = StubEmbedder(dim=768)
    ids = _seed(tmp_db, stub)
    hits = search_items(tmp_db, "boron dilution", mode="hybrid", limit=5, item_embedder=stub)
    # Hybrid should return the boron paper first (keyword strongly matches).
    assert hits[0].item_id == ids[0]
    assert all(h.score > 0 for h in hits)  # RRF scores are strictly positive


def test_limit_respected(tmp_db: sqlite3.Connection) -> None:
    stub = StubEmbedder(dim=768)
    _seed(tmp_db, stub)
    hits = search_items(tmp_db, "a", mode="hybrid", limit=2, item_embedder=stub)
    assert len(hits) <= 2
