"""Crossref DOI resolver via habanero. Polite pool ``mailto`` is read from config."""

from __future__ import annotations

import logging
import re
from typing import Any

from grimoire.config import settings
from grimoire.models import Author, Metadata

log = logging.getLogger(__name__)

_HTML_TAG = re.compile(r"<[^>]+>")
_JATS_TITLE = re.compile(r"<jats:title[^>]*>[^<]*</jats:title>")


def _fetch_raw(doi: str) -> dict[str, Any] | None:
    """Hit Crossref for a DOI. Returns the full ``works`` payload or None on miss.

    Isolated so tests can monkeypatch without importing habanero."""
    try:
        from habanero import Crossref
    except ImportError:  # pragma: no cover - habanero is in the ingest extra
        log.warning("habanero not installed; Crossref resolution disabled")
        return None

    cr = Crossref(mailto=settings.crossref_mailto) if settings.crossref_mailto else Crossref()
    try:
        return cr.works(ids=doi)  # type: ignore[no-any-return]
    except Exception as exc:
        log.warning("Crossref lookup failed for %s: %s", doi, exc)
        return None


def resolve(doi: str) -> Metadata | None:
    raw = _fetch_raw(doi)
    if not raw:
        return None
    msg = raw.get("message") if isinstance(raw, dict) else None
    if not msg:
        return None
    return _to_metadata(msg)


def _to_metadata(msg: dict[str, Any]) -> Metadata:
    title = _first(msg.get("title"))
    abstract_raw = msg.get("abstract")
    abstract = _strip_html(_JATS_TITLE.sub("", abstract_raw)) if abstract_raw else None
    year = _year_from_issued(msg.get("issued"))

    authors = [
        Author(
            family_name=(a.get("family") or "").strip(),
            given_name=(a.get("given") or None),
            orcid=_extract_orcid(a.get("ORCID")),
        )
        for a in msg.get("author", [])
        if a.get("family")
    ]

    item_type = _map_type(msg.get("type"))

    return Metadata(
        title=title,
        abstract=abstract,
        publication_year=year,
        doi=msg.get("DOI"),
        venue=_first(msg.get("container-title")),
        volume=msg.get("volume"),
        issue=msg.get("issue"),
        pages=msg.get("page"),
        edition=_edition(msg),
        language=msg.get("language"),
        item_type=item_type,
        authors=authors,
        source="crossref",
        confidence=1.0,
        raw={"crossref": msg},
    )


def _edition(msg: dict[str, Any]) -> str | None:
    """Crossref ships edition under several keys across product lines. Prefer
    the explicit ``edition-number`` (usually a bare integer), then
    ``edition``, then the ``edition_number`` alternate."""
    for key in ("edition-number", "edition_number", "edition"):
        val = msg.get(key)
        if isinstance(val, (str, int)) and str(val).strip():
            return str(val).strip()
    return None


def _first(xs: Any) -> str | None:
    if isinstance(xs, list) and xs:
        value = xs[0]
        return str(value) if value is not None else None
    return None


def _strip_html(s: str) -> str:
    return _HTML_TAG.sub("", s).strip()


def _year_from_issued(issued: Any) -> int | None:
    if not isinstance(issued, dict):
        return None
    parts = issued.get("date-parts")
    if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
        return int(parts[0][0])
    return None


def _extract_orcid(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", url)
    return m.group(1) if m else None


# Crossref type → our item_type vocabulary. Unknown types → 'paper' for safety.
_TYPE_MAP = {
    "journal-article": "paper",
    "proceedings-article": "paper",
    "posted-content": "preprint",
    "book": "book",
    "book-chapter": "chapter",
    "edited-book": "book",
    "monograph": "book",
    "report": "report",
    "dissertation": "thesis",
    "standard": "standard",
}


def _map_type(t: Any) -> str:
    return _TYPE_MAP.get(t, "paper") if isinstance(t, str) else "paper"
