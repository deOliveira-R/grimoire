from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Snippet:
    chunk_id: int
    item_id: int
    page: int | None
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class ItemHit:
    item_id: int
    score: float


@dataclass(frozen=True, slots=True)
class SearchHit:
    item_id: int
    score: float
    title: str
    year: int | None
    snippet: Snippet | None
