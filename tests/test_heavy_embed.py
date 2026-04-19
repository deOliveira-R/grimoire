"""Real-model integration tests. Marked `heavy` so they stay out of CI runs.

Run with:
    pytest -m heavy

These download SPECTER2 (~440MB) and BGE-M3 (~2.27GB) to the Hugging Face
cache on first run, then re-use them on subsequent runs."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from grimoire.embed.base import l2_normalize

pytestmark = pytest.mark.heavy


def test_specter2_encodes_to_768_dim() -> None:
    from grimoire.embed.specter2 import DIM, Specter2Embedder

    emb = Specter2Embedder()
    v = emb.encode(["A paper [SEP] with an abstract."])
    assert v.shape == (1, DIM) == (1, 768)
    assert v.dtype == np.float32
    assert np.isfinite(v).all()


def test_bge_m3_encodes_to_1024_dim() -> None:
    from grimoire.embed.bge_m3 import DIM, BGEM3Embedder

    emb = BGEM3Embedder()
    v = emb.encode(["fuel rod thermal analysis"])
    assert v.shape == (1, DIM) == (1, 1024)
    assert v.dtype == np.float32
    assert np.isfinite(v).all()


def test_specter2_cold_query_under_500ms(tmp_db: sqlite3.Connection, pdf_with_doi: Path) -> None:
    """Plan §6 Phase 2 oracle: cold-start query latency < 500ms AFTER model is loaded."""
    from grimoire.embed.specter2 import Specter2Embedder
    from grimoire.search.semantic import search_items_by_embedding

    emb = Specter2Embedder()
    _ = emb.encode(["warm up"])  # materialize weights + first-call kernels

    # Seed a few items with real embeddings so the index isn't empty.
    pairs = [
        ("Boron dilution transients in PWR reactors", "boron concentration study"),
        ("Fuel rod thermal modeling", "heat transfer analysis"),
        ("Vegetable recipes", "cooking tips"),
    ]
    vecs = l2_normalize(emb.encode([f"{t} [SEP] {a}" for t, a in pairs]))
    for (title, _abstract), vec in zip(pairs, vecs, strict=True):
        row = tmp_db.execute(
            "INSERT INTO items(item_type, title) VALUES ('paper', ?) RETURNING id", (title,)
        ).fetchone()
        from grimoire.embed.base import serialize_float32

        tmp_db.execute(
            "INSERT INTO item_embeddings(item_id, embedding) VALUES (?, ?)",
            (row["id"], serialize_float32(vec)),
        )

    # Time: embed query + vec0 KNN.
    t0 = time.perf_counter()
    q = l2_normalize(emb.encode(["fuel rod heat transfer"]))[0]
    hits = search_items_by_embedding(tmp_db, q, limit=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert hits, "expected at least one hit"
    # Hardware target is the X9SCM-F; on any modern laptop the number is tighter.
    assert elapsed_ms < 500, f"cold-cache query took {elapsed_ms:.1f}ms (plan target: <500ms)"


def test_bge_m3_semantic_beats_random(tmp_db: sqlite3.Connection) -> None:
    """Minimal sanity check: a topically-related chunk should rank above an unrelated one."""
    from grimoire.embed.base import serialize_float32
    from grimoire.embed.bge_m3 import BGEM3Embedder
    from grimoire.search.semantic import search_chunks_by_embedding

    emb = BGEM3Embedder()

    related = "Boron dilution transients cause reactivity excursions in PWR reactors."
    unrelated = "This tart is best served warm with vanilla ice cream."

    item_id = tmp_db.execute(
        "INSERT INTO items(item_type, title) VALUES ('paper','p') RETURNING id"
    ).fetchone()["id"]

    texts = [related, unrelated]
    vecs = l2_normalize(emb.encode(texts))
    for i, (text, vec) in enumerate(zip(texts, vecs, strict=True)):
        cur = tmp_db.execute(
            "INSERT INTO chunks(item_id, chunk_index, page, text) VALUES (?, ?, ?, ?)",
            (item_id, i, 1, text),
        )
        tmp_db.execute(
            "INSERT INTO chunk_embeddings(chunk_id, embedding) VALUES (?, ?)",
            (int(cur.lastrowid), serialize_float32(vec)),  # type: ignore[arg-type]
        )

    q = l2_normalize(emb.encode(["boron dilution in PWR"]))[0]
    hits = search_chunks_by_embedding(tmp_db, q, limit=2)
    assert hits[0].text == related, f"expected boron chunk first, got: {hits[0].text[:60]!r}"
