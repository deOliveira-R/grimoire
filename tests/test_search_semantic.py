from __future__ import annotations

import sqlite3

import numpy as np

from grimoire.embed.base import l2_normalize, serialize_float32
from grimoire.search.semantic import search_chunks_by_embedding, search_items_by_embedding


def _insert_item_with_embedding(conn: sqlite3.Connection, title: str, embedding: np.ndarray) -> int:
    cur = conn.execute("INSERT INTO items(item_type, title) VALUES ('paper', ?)", (title,))
    item_id = int(cur.lastrowid)  # type: ignore[arg-type]
    conn.execute(
        "INSERT INTO item_embeddings(item_id, embedding) VALUES (?, ?)",
        (item_id, serialize_float32(embedding)),
    )
    return item_id


def _insert_chunk_with_embedding(
    conn: sqlite3.Connection, item_id: int, idx: int, page: int, text: str, embedding: np.ndarray
) -> int:
    cur = conn.execute(
        "INSERT INTO chunks(item_id, chunk_index, page, text) VALUES (?, ?, ?, ?)",
        (item_id, idx, page, text),
    )
    chunk_id = int(cur.lastrowid)  # type: ignore[arg-type]
    conn.execute(
        "INSERT INTO chunk_embeddings(chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_float32(embedding)),
    )
    return chunk_id


def test_item_semantic_ranks_by_proximity(tmp_db: sqlite3.Connection) -> None:
    rng = np.random.default_rng(0)
    target = l2_normalize(rng.standard_normal((1, 768)).astype(np.float32))[0]
    near = l2_normalize(
        (target + 0.01 * rng.standard_normal(768)).reshape(1, -1).astype(np.float32)
    )[0]
    far = l2_normalize(rng.standard_normal((1, 768)).astype(np.float32))[0]

    a = _insert_item_with_embedding(tmp_db, "far", far)
    b = _insert_item_with_embedding(tmp_db, "near", near)

    hits = search_items_by_embedding(tmp_db, target, limit=5)
    assert hits[0].item_id == b
    assert hits[-1].item_id == a


def test_chunk_semantic_ranks_by_proximity(tmp_db: sqlite3.Connection) -> None:
    rng = np.random.default_rng(1)
    target = l2_normalize(rng.standard_normal((1, 1024)).astype(np.float32))[0]
    near = l2_normalize(
        (target + 0.005 * rng.standard_normal(1024)).reshape(1, -1).astype(np.float32)
    )[0]
    far = l2_normalize(rng.standard_normal((1, 1024)).astype(np.float32))[0]

    item_id = tmp_db.execute(
        "INSERT INTO items(item_type, title) VALUES ('paper','p') RETURNING id"
    ).fetchone()["id"]
    near_id = _insert_chunk_with_embedding(tmp_db, item_id, 0, 1, "relevant text", near)
    _insert_chunk_with_embedding(tmp_db, item_id, 1, 2, "other text", far)

    hits = search_chunks_by_embedding(tmp_db, target, limit=5)
    assert hits[0].chunk_id == near_id
    assert hits[0].page == 1
    assert "relevant" in hits[0].text


def test_empty_index_returns_empty(tmp_db: sqlite3.Connection) -> None:
    target = np.zeros(768, dtype=np.float32)
    target[0] = 1.0
    assert search_items_by_embedding(tmp_db, target, limit=10) == []
