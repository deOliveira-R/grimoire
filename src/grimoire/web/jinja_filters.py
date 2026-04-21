"""Custom Jinja2 filters for the web UI."""

from __future__ import annotations

import re

from markupsafe import Markup, escape

from grimoire.search.keyword import _TOKEN


def highlight(text: str | None, query: str | None) -> Markup:
    """Wrap query terms in ``<mark>``. Case-insensitive.

    HTML-escapes the input first, then injects ``<mark>`` around matches
    so the result is safe to render without further escaping. Returning
    a ``Markup`` prevents Jinja from re-escaping."""
    if not text:
        return Markup(escape(text or ""))
    if not query:
        return Markup(escape(text))
    tokens = _TOKEN.findall(query)
    if not tokens:
        return Markup(escape(text))
    pattern = re.compile(
        "(" + "|".join(re.escape(t) for t in tokens) + ")",
        re.IGNORECASE,
    )
    escaped = str(escape(text))
    return Markup(pattern.sub(r"<mark>\1</mark>", escaped))
