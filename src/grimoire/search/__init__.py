"""Search facade: keyword / semantic / hybrid. Returns ranked items with
optional best-matching snippets."""

from __future__ import annotations

import sqlite3
from typing import Literal

from grimoire.embed.base import Embedder, l2_normalize
from grimoire.embed.specter2 import format_item_text
from grimoire.search import keyword, semantic
from grimoire.search.hybrid import reciprocal_rank_fusion
from grimoire.search.models import SearchHit, Snippet

SearchMode = Literal["hybrid", "keyword", "semantic"]

# RRF constant from plan §6 Phase 2. Chosen to follow the Cormack et al. paper.
RRF_K = 60


def search_items(
    conn: sqlite3.Connection,
    query: str,
    mode: SearchMode = "hybrid",
    limit: int = 20,
    item_embedder: Embedder | None = None,
    chunk_embedder: Embedder | None = None,
) -> list[SearchHit]:
    if not query.strip():
        return []

    if mode == "keyword":
        hits = keyword.search_items(conn, query, limit=limit)
        ranked_ids = [h.item_id for h in hits]
    elif mode == "semantic":
        if item_embedder is None:
            raise ValueError("semantic mode requires item_embedder")
        q_vec = l2_normalize(item_embedder.encode([_query_for_item_embed(query)]))[0]
        hits = semantic.search_items_by_embedding(conn, q_vec, limit=limit)
        ranked_ids = [h.item_id for h in hits]
    elif mode == "hybrid":
        if item_embedder is None:
            raise ValueError("hybrid mode requires item_embedder")
        # Broader candidate pool before fusion, then trim to `limit`.
        pool = max(limit * 3, 60)
        kw = [h.item_id for h in keyword.search_items(conn, query, limit=pool)]
        q_vec = l2_normalize(item_embedder.encode([_query_for_item_embed(query)]))[0]
        sem = [h.item_id for h in semantic.search_items_by_embedding(conn, q_vec, limit=pool)]
        fused = reciprocal_rank_fusion([kw, sem], k=RRF_K)[:limit]
        ranked_ids = [item_id for item_id, _ in fused]
    else:  # pragma: no cover - type-checker catches this already
        raise ValueError(f"unknown mode: {mode}")

    return _hydrate(conn, query, ranked_ids, mode, chunk_embedder)


def _query_for_item_embed(query: str) -> str:
    """Format the user's query as SPECTER2 expects: ``title [SEP] abstract``.
    We put the query in the title slot so shorter queries still pick up the
    domain-pretrained signal."""
    return format_item_text(query, abstract=None)


def _hydrate(
    conn: sqlite3.Connection,
    query: str,
    item_ids: list[int],
    mode: SearchMode,
    chunk_embedder: Embedder | None,
) -> list[SearchHit]:
    if not item_ids:
        return []

    placeholders = ",".join("?" * len(item_ids))
    rows = conn.execute(
        f"SELECT id, title, publication_year FROM items WHERE id IN ({placeholders})",
        item_ids,
    ).fetchall()
    by_id = {row["id"]: row for row in rows}

    snippets = _best_snippets(conn, query, item_ids, mode, chunk_embedder)

    out = []
    for rank, item_id in enumerate(item_ids, start=1):
        row = by_id.get(item_id)
        if row is None:
            continue
        # Rank is 1-based; score is 1/rank so that hits compare ordinally.
        out.append(
            SearchHit(
                item_id=int(item_id),
                score=1.0 / rank,
                title=row["title"],
                year=row["publication_year"],
                snippet=snippets.get(item_id),
            )
        )
    return out


def _best_snippets(
    conn: sqlite3.Connection,
    query: str,
    item_ids: list[int],
    mode: SearchMode,
    chunk_embedder: Embedder | None,
) -> dict[int, Snippet]:
    """For each item, return the single best-matching chunk. For keyword mode
    we run FTS5 over chunks_fts; for semantic/hybrid we use the chunk embedder
    when one is provided; otherwise we fall back to FTS5."""
    if not item_ids:
        return {}

    if mode == "semantic" and chunk_embedder is not None:
        q_vec = l2_normalize(chunk_embedder.encode([query]))[0]
        hits = semantic.search_chunks_by_embedding(conn, q_vec, limit=len(item_ids) * 5)
        return _first_per_item(hits, item_ids)

    if mode == "hybrid" and chunk_embedder is not None:
        q_vec = l2_normalize(chunk_embedder.encode([query]))[0]
        sem_hits = semantic.search_chunks_by_embedding(conn, q_vec, limit=len(item_ids) * 5)
        if sem_hits:
            return _first_per_item(sem_hits, item_ids)

    # Fallback: FTS5 over chunks.
    kw_hits = keyword.search_chunks(conn, query, limit=len(item_ids) * 5)
    return _first_per_item(kw_hits, item_ids)


def _first_per_item(hits: list[Snippet], item_ids: list[int]) -> dict[int, Snippet]:
    wanted = set(item_ids)
    out: dict[int, Snippet] = {}
    for h in hits:
        if h.item_id in wanted and h.item_id not in out:
            out[h.item_id] = h
    return out


__all__ = ["RRF_K", "SearchHit", "SearchMode", "Snippet", "search_items"]
