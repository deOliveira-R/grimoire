"""PyMuPDF-based text extraction. OCR (plan §8) is opt-in and not wired here."""

from __future__ import annotations

from pathlib import Path

import pymupdf


def extract_text(path: Path) -> str:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def extract_first_page(path: Path) -> str:
    with pymupdf.open(path) as doc:
        if doc.page_count == 0:
            return ""
        return doc.load_page(0).get_text()


def page_count(path: Path) -> int:
    with pymupdf.open(path) as doc:
        return doc.page_count


def has_extractable_text(path: Path) -> bool:
    return bool(extract_first_page(path).strip())
