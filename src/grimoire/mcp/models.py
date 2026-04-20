"""Pydantic models exposed over MCP. Kept small and JSON-serializable —
MCP clients see these as tool-result schemas."""

from __future__ import annotations

from pydantic import BaseModel


class Snippet(BaseModel):
    item_id: int
    chunk_id: int
    page: int | None
    chunk_index: int
    text: str
    score: float


class ItemSummary(BaseModel):
    item_id: int
    title: str
    year: int | None = None
    authors: list[str] = []  # last names, ordered
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    item_type: str = "paper"
    score: float | None = None
    snippet: Snippet | None = None  # best-matching chunk when produced by search


class ItemFull(ItemSummary):
    abstract: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    series: str | None = None
    edition: str | None = None
    isbn: str | None = None
    language: str | None = None
    tags: list[str] = []
    collections: list[str] = []
    metadata_source: str | None = None
    metadata_confidence: float | None = None


class RelatedItem(ItemSummary):
    relation: str  # e.g. 'preprint_of', 'related', 'erratum_for'
    confidence: float = 1.0


class Collection(BaseModel):
    id: int
    name: str
    parent_id: int | None = None
    item_count: int = 0
