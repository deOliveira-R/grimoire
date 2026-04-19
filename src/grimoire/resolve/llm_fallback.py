"""Last-resort metadata extraction: hand the PDF's first page to Claude and
ask for structured metadata. Only runs if ANTHROPIC_API_KEY is configured.

Returns None when the key isn't set — ingest then falls back to manual_required
(oracle check 3)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from grimoire.config import settings
from grimoire.models import Author, Metadata

log = logging.getLogger(__name__)

_SYSTEM = """You extract bibliographic metadata from the first page of a paper or book.
Return ONLY a JSON object with these keys (null where unknown):
  title: string
  abstract: string
  publication_year: integer
  authors: [{family: string, given: string|null}]
  venue: string
  doi: string
  arxiv_id: string
  isbn: string
Do not invent values. If the page shows a clear identifier, include it.
"""


def resolve(first_page_text: str) -> Metadata | None:
    if not settings.anthropic_api_key:
        return None
    if not first_page_text.strip():
        return None

    try:
        import anthropic
    except ImportError:  # pragma: no cover
        log.warning("anthropic package not installed; LLM fallback disabled")
        return None

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        message = client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": first_page_text[:8000]}],
        )
    except Exception as exc:
        log.warning("LLM fallback call failed: %s", exc)
        return None

    text = "".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text" and hasattr(block, "text")
    )
    parsed = _parse_json(text)
    if parsed is None:
        return None
    return _to_metadata(parsed)


def _parse_json(text: str) -> dict[str, Any] | None:
    # Tolerate surrounding prose; pull the first JSON object we see.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        result = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def _to_metadata(d: dict[str, Any]) -> Metadata:
    authors = []
    for a in d.get("authors") or []:
        if isinstance(a, dict) and a.get("family"):
            authors.append(Author(family_name=str(a["family"]), given_name=a.get("given") or None))
    return Metadata(
        title=d.get("title") or None,
        abstract=d.get("abstract") or None,
        publication_year=_as_int(d.get("publication_year")),
        doi=d.get("doi") or None,
        arxiv_id=d.get("arxiv_id") or None,
        isbn=d.get("isbn") or None,
        venue=d.get("venue") or None,
        authors=authors,
        source="llm",
        confidence=0.7,
        raw={"llm": d},
    )


def _as_int(x: Any) -> int | None:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None
