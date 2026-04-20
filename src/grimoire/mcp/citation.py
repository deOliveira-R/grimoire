"""BibTeX generator. Plan §9 defers CSL styles (APA/IEEE/Nature) to v2."""

from __future__ import annotations

import re
import sqlite3

from grimoire.mcp.tools import get_item

_TYPE_TO_BIBTEX = {
    "paper": "article",
    "preprint": "article",
    "book": "book",
    "chapter": "incollection",
    "report": "techreport",
    "thesis": "phdthesis",
    "standard": "misc",
    "patent": "misc",
    "other": "misc",
}

# Characters that need escaping in a BibTeX field value.
_ESCAPE = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$"}


def to_bibtex(conn: sqlite3.Connection, item_id: int) -> str | None:
    item = get_item(conn, item_id)
    if item is None:
        return None

    entry_type = _TYPE_TO_BIBTEX.get(item.item_type, "misc")
    key = _bibtex_key(item.authors, item.year, item.title)

    fields: list[tuple[str, str]] = []
    if item.title:
        # Double-brace titles so BibTeX preserves capitalization.
        fields.append(("title", "{" + _escape(item.title) + "}"))
    if item.authors:
        # BibTeX "Last, First and Last, First" form. We only have family names;
        # rendering as bare surnames is acceptable.
        fields.append(("author", _escape(" and ".join(item.authors))))
    if item.year is not None:
        fields.append(("year", str(item.year)))
    if item.venue:
        field = "booktitle" if entry_type in {"incollection", "inproceedings"} else "journal"
        fields.append((field, _escape(item.venue)))
    if item.volume:
        fields.append(("volume", _escape(item.volume)))
    if item.issue:
        fields.append(("number", _escape(item.issue)))
    if item.pages:
        fields.append(("pages", _escape(item.pages)))
    if item.doi:
        fields.append(("doi", _escape(item.doi)))
    if item.isbn:
        fields.append(("isbn", _escape(item.isbn)))
    if item.arxiv_id:
        fields.append(("eprint", _escape(item.arxiv_id)))
        fields.append(("archivePrefix", "arXiv"))
    if item.edition:
        fields.append(("edition", _escape(item.edition)))
    if item.series:
        fields.append(("series", _escape(item.series)))
    if item.language and item.language != "en":
        fields.append(("language", _escape(item.language)))

    body = ",\n".join(
        f"  {k} = {{{v}}}" if not v.startswith("{") else f"  {k} = {v}" for k, v in fields
    )
    return f"@{entry_type}{{{key},\n{body}\n}}"


def _escape(text: str) -> str:
    out = text
    for src, dst in _ESCAPE.items():
        out = out.replace(src, dst)
    return out


def _bibtex_key(authors: list[str], year: int | None, title: str | None) -> str:
    parts = []
    if authors:
        parts.append(_asciify(authors[0]).lower())
    if year is not None:
        parts.append(str(year))
    if title:
        # First content word of the title, <= 12 chars
        words = re.findall(r"[A-Za-z0-9]+", _asciify(title))
        for w in words:
            if len(w) > 2 and w.lower() not in {"the", "and", "for", "from", "with", "into"}:
                parts.append(w.lower()[:12])
                break
    if not parts:
        parts.append("ref")
    return "_".join(parts) or "ref"


def _asciify(s: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", s)
    return "".join(c for c in normalized if not unicodedata.combining(c))
