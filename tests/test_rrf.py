from __future__ import annotations

from grimoire.search.hybrid import reciprocal_rank_fusion


def test_rrf_single_ranking_preserves_order() -> None:
    fused = reciprocal_rank_fusion([[10, 20, 30]], k=60)
    assert [item for item, _ in fused] == [10, 20, 30]


def test_rrf_combines_two_rankings() -> None:
    # Item 2 is top in the second ranking and near-top in the first, so it
    # should beat item 1 (only top in one) and item 4 (only bottom in one).
    keyword = [1, 2, 3]
    semantic = [2, 3, 4]
    fused = reciprocal_rank_fusion([keyword, semantic], k=60)
    ids = [item for item, _ in fused]
    assert ids[0] == 2
    assert ids.index(3) < ids.index(4)  # 3 appears in both, 4 only once


def test_rrf_boosts_intersection() -> None:
    a = [1, 2, 3, 4, 5]
    b = [1, 99, 98, 97, 96]  # 1 is top in both
    fused = reciprocal_rank_fusion([a, b], k=60)
    ids = [item for item, _ in fused]
    assert ids[0] == 1


def test_rrf_k_parameter_matters() -> None:
    a = [1, 2]
    b = [2, 1]
    k1 = reciprocal_rank_fusion([a, b], k=1)
    k60 = reciprocal_rank_fusion([a, b], k=60)
    # Scores differ; ordering for symmetric case is a tie either way.
    assert dict(k1) != dict(k60)


def test_rrf_empty() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[]], k=60) == []
