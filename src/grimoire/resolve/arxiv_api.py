"""arXiv resolver. The linked DOI (when arXiv has one) is stored in raw so the
ingest pipeline can later create a ``preprint_of`` relation — see plan §3 tier 3."""

from __future__ import annotations

import logging
from typing import Any

from grimoire.models import Author, Metadata

log = logging.getLogger(__name__)


def _fetch_raw(arxiv_id: str) -> dict[str, Any] | None:
    try:
        import arxiv
    except ImportError:  # pragma: no cover
        log.warning("arxiv package not installed; arXiv resolution disabled")
        return None

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(arxiv.Client().results(search), None)
    except Exception as exc:
        log.warning("arXiv lookup failed for %s: %s", arxiv_id, exc)
        return None
    if result is None:
        return None

    return {
        "entry_id": result.entry_id,
        "title": result.title,
        "summary": result.summary,
        "published_year": result.published.year if result.published else None,
        "authors": [{"name": a.name} for a in result.authors],
        "doi": result.doi,
        "journal_ref": result.journal_ref,
    }


def resolve(arxiv_id: str) -> Metadata | None:
    raw = _fetch_raw(arxiv_id)
    if not raw:
        return None
    return _to_metadata(arxiv_id, raw)


def _to_metadata(arxiv_id: str, raw: dict[str, Any]) -> Metadata:
    authors = [_parse_author(a.get("name") or "") for a in raw.get("authors", [])]
    authors = [a for a in authors if a.family_name]

    meta_raw: dict[str, Any] = {"arxiv": raw}
    if raw.get("doi"):
        # Tier-3 signal for preprint→published relation (plan §3).
        meta_raw["linked_doi"] = raw["doi"]

    return Metadata(
        title=(raw.get("title") or "").strip(),
        abstract=(raw.get("summary") or "").strip() or None,
        publication_year=raw.get("published_year"),
        arxiv_id=arxiv_id,
        venue=raw.get("journal_ref"),
        item_type="preprint",
        authors=authors,
        source="arxiv",
        confidence=1.0,
        raw=meta_raw,
    )


def _parse_author(full_name: str) -> Author:
    parts = full_name.strip().split()
    if not parts:
        return Author(family_name="", given_name=None)
    if len(parts) == 1:
        return Author(family_name=parts[0], given_name=None)
    return Author(family_name=parts[-1], given_name=" ".join(parts[:-1]))
