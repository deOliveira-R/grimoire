"""EPUB text + metadata extraction via ebooklib + BeautifulSoup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub


def _read(path: Path) -> epub.EpubBook:
    return epub.read_epub(str(path))


def extract_text(path: Path) -> str:
    book = _read(path)
    parts: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        parts.append(soup.get_text(separator="\n"))
    return "\n".join(parts)


def extract_metadata(path: Path) -> dict[str, Any]:
    book = _read(path)

    def first(name: str) -> str | None:
        values = book.get_metadata("DC", name)
        return values[0][0] if values else None

    def all_of(name: str) -> list[str]:
        return [v[0] for v in book.get_metadata("DC", name)]

    return {
        "title": first("title"),
        "creators": all_of("creator"),
        "language": first("language"),
        "publisher": first("publisher"),
        "date": first("date"),
        "identifiers": all_of("identifier"),
    }
