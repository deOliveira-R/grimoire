"""Reciprocal Rank Fusion. Plan §6 Phase 2 specifies k=60 per Cormack et al."""

from __future__ import annotations

from collections.abc import Iterable


def reciprocal_rank_fusion(rankings: Iterable[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse any number of id-rankings into a single ranked list.

    Each input is a list of item ids ordered from best to worst. The fused
    score for an id is ``sum(1 / (k + rank))`` over all rankings that include
    it, where rank is 1-based."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
