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
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    limit: int = 20,
    *,
    section: str | None = None,
) -> list[Snippet]:
    # ``section`` is applied as a post-KNN filter. vec0's MATCH + k preselects
    # before the JOIN, so we over-request when filtering to preserve recall.
    k = limit if section is None else limit * 5
    rows = conn.execute(
        """SELECT ce.chunk_id, ce.distance,
                  c.item_id, c.page, c.text, c.section
           FROM chunk_embeddings ce
           JOIN chunks c ON c.id = ce.chunk_id
           WHERE ce.embedding MATCH ?
             AND ce.k = ?
           ORDER BY ce.distance""",
        (serialize_float32(query_vec), k),
    ).fetchall()
    out: list[Snippet] = []
    for r in rows:
        if section is not None and r["section"] != section:
            continue
        out.append(
            Snippet(
                chunk_id=int(r["chunk_id"]),
                item_id=int(r["item_id"]),
                page=r["page"],
                text=r["text"],
                score=-float(r["distance"]),
            )
        )
        if len(out) >= limit:
            break
    return out
