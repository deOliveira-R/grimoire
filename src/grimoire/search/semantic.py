"""sqlite-vec KNN queries. Vectors are stored L2-normalized so default L2
distance preserves cosine ranking — see grimoire.embed.base."""

from __future__ import annotations

import sqlite3

import numpy as np

from grimoire.embed.base import serialize_float32
from grimoire.search.models import ItemHit, Snippet


def search_items_by_embedding(
    conn: sqlite3.Connection, query_vec: np.ndarray, limit: int = 20
) -> list[ItemHit]:
    rows = conn.execute(
        """SELECT item_id, distance
           FROM item_embeddings
           WHERE embedding MATCH ?
             AND k = ?
           ORDER BY distance""",
        (serialize_float32(query_vec), limit),
    ).fetchall()
    return [ItemHit(item_id=int(r["item_id"]), score=-float(r["distance"])) for r in rows]


def search_chunks_by_embedding(
    conn: sqlite3.Connection, query_vec: np.ndarray, limit: int = 20
) -> list[Snippet]:
    rows = conn.execute(
        """SELECT ce.chunk_id, ce.distance,
                  c.item_id, c.page, c.text
           FROM chunk_embeddings ce
           JOIN chunks c ON c.id = ce.chunk_id
           WHERE ce.embedding MATCH ?
             AND ce.k = ?
           ORDER BY ce.distance""",
        (serialize_float32(query_vec), limit),
    ).fetchall()
    return [
        Snippet(
            chunk_id=int(r["chunk_id"]),
            item_id=int(r["item_id"]),
            page=r["page"],
            text=r["text"],
            score=-float(r["distance"]),
        )
        for r in rows
    ]
