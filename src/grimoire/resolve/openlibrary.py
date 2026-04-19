"""OpenLibrary ISBN resolver via their Books API."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from grimoire.identify import normalize_isbn
from grimoire.models import Author, Metadata

log = logging.getLogger(__name__)

_BASE = "https://openlibrary.org/api/books"


def _fetch_raw(isbn: str) -> dict[str, Any] | None:
    params = {"bibkeys": f"ISBN:{isbn}", "jscmd": "data", "format": "json"}
    try:
        r = httpx.get(_BASE, params=params, timeout=15.0)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        log.warning("OpenLibrary lookup failed for %s: %s", isbn, exc)
        return None
    result = payload.get(f"ISBN:{isbn}")
    return result if isinstance(result, dict) else None


def resolve(isbn: str) -> Metadata | None:
    normalized = normalize_isbn(isbn)
    raw = _fetch_raw(normalized)
    if not raw:
        return None
    return _to_metadata(normalized, raw)


def _to_metadata(isbn: str, raw: dict[str, Any]) -> Metadata:
    authors = [_parse_author(a.get("name") or "") for a in raw.get("authors", [])]
    authors = [a for a in authors if a.family_name]

    return Metadata(
        title=raw.get("title"),
        publication_year=_year(raw.get("publish_date")),
        isbn=isbn,
        venue=_first_publisher(raw),
        item_type="book",
        authors=authors,
        source="openlibrary",
        confidence=0.9,
        raw={"openlibrary": raw},
    )


def _year(raw: Any) -> int | None:
    if not isinstance(raw, str):
        return None
    m = re.search(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b", raw)
    return int(m.group(1)) if m else None


def _first_publisher(raw: dict[str, Any]) -> str | None:
    pubs = raw.get("publishers")
    if isinstance(pubs, list) and pubs:
        first = pubs[0]
        if isinstance(first, dict):
            return first.get("name")
        if isinstance(first, str):
            return first
    return None


def _parse_author(full: str) -> Author:
    parts = full.strip().split()
    if not parts:
        return Author(family_name="", given_name=None)
    if len(parts) == 1:
        return Author(family_name=parts[0], given_name=None)
    return Author(family_name=parts[-1], given_name=" ".join(parts[:-1]))
