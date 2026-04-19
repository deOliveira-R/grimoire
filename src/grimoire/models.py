"""Domain dataclasses used across resolvers and the ingest pipeline.

These are deliberately narrow: they carry only the fields the rest of the system
touches. The full raw payload from each upstream source is preserved in
``Metadata.raw`` and persisted to ``items.metadata_json``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SourceName = Literal[
    "crossref",
    "arxiv",
    "openlibrary",
    "llm",
    "zotero_import",
    "manual",
    "manual_required",
]

# Plan §3 merge semantics: Crossref > arXiv > OpenLibrary > LLM.
_SOURCE_RANK: dict[str, int] = {
    "crossref": 4,
    "arxiv": 3,
    "openlibrary": 2,
    "llm": 1,
    "zotero_import": 0,
    "manual": 0,
    "manual_required": -1,
}


@dataclass(frozen=True, slots=True)
class Author:
    family_name: str
    given_name: str | None = None
    orcid: str | None = None

    @property
    def normalized_key(self) -> str:
        from grimoire.identify import normalize_author_key

        return normalize_author_key(self.family_name, self.given_name)


@dataclass(slots=True)
class Metadata:
    title: str | None = None
    abstract: str | None = None
    publication_year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    isbn: str | None = None
    venue: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    series: str | None = None
    series_number: str | None = None
    edition: str | None = None
    language: str | None = None
    item_type: str = "paper"
    authors: list[Author] = field(default_factory=list)
    source: SourceName = "manual_required"
    confidence: float = 0.0
    raw: dict[str, Any] | None = None


def prefer_more_authoritative(candidates: list[Metadata]) -> Metadata:
    """Return the Metadata with the highest-ranked source; break ties by confidence."""
    return max(candidates, key=lambda m: (_SOURCE_RANK.get(m.source, -2), m.confidence))


@dataclass(slots=True)
class IngestResult:
    outcome: Literal["inserted", "skipped", "merged", "failed"]
    item_id: int | None
    source_path: str
    content_hash: str | None = None
    reason: str | None = None
