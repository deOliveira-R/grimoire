"""Identifier extraction from raw text (first-page of a PDF, EPUB frontmatter, etc).

All extractors return de-duplicated, normalized candidates. Tests in
``tests/test_identify.py`` pin the behaviour."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# DOI per Crossref guidance: 10.<prefix>/<suffix>. Suffix is permissive; we strip
# trailing punctuation that is almost never part of the DOI itself.
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>{}|\\^`\[\]]+)", re.IGNORECASE)
_DOI_TRAILING = re.compile(r"[.,;:)\]]+$")

# arXiv identifiers: new-style (post-2007) and old-style (pre-2007).
_ARXIV_NEW = re.compile(r"arxiv[:\s]*?(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_ARXIV_OLD = re.compile(r"arxiv[:\s]*?([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE)
_ARXIV_URL_NEW = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_ARXIV_URL_OLD = re.compile(
    r"arxiv\.org/abs/([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE
)

# ISBN: only match when preceded by the "ISBN" keyword. Random 10/13-digit
# runs show up in phone numbers and catalog numbers too often otherwise.
_ISBN_RE = re.compile(
    r"ISBN(?:-1[03])?[:\s]*([0-9][\d\-\sXx]{8,20}[0-9Xx])",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Identifiers:
    dois: list[str]
    arxiv_ids: list[str]
    isbns: list[str]


def identify(text: str) -> Identifiers:
    return Identifiers(
        dois=extract_dois(text),
        arxiv_ids=extract_arxiv_ids(text),
        isbns=extract_isbns(text),
    )


def extract_dois(text: str) -> list[str]:
    seen: list[str] = []
    for m in _DOI_RE.finditer(text):
        candidate = _DOI_TRAILING.sub("", m.group(1))
        if candidate not in seen:
            seen.append(candidate)
    return seen


def extract_arxiv_ids(text: str) -> list[str]:
    seen: list[str] = []
    for pattern in (_ARXIV_URL_NEW, _ARXIV_URL_OLD, _ARXIV_NEW, _ARXIV_OLD):
        for m in pattern.finditer(text):
            candidate = m.group(1)
            if candidate not in seen:
                seen.append(candidate)
    return seen


def extract_isbns(text: str) -> list[str]:
    seen: list[str] = []
    for m in _ISBN_RE.finditer(text):
        raw = re.sub(r"[\s\-]", "", m.group(1)).upper()
        if not is_valid_isbn(raw):
            continue
        normalized = normalize_isbn(raw)
        if normalized not in seen:
            seen.append(normalized)
    return seen


def is_valid_isbn(raw: str) -> bool:
    digits = re.sub(r"[\s\-]", "", raw).upper()
    if len(digits) == 10:
        return _isbn10_checksum(digits)
    if len(digits) == 13:
        return _isbn13_checksum(digits)
    return False


def normalize_isbn(raw: str) -> str:
    digits = re.sub(r"[\s\-]", "", raw).upper()
    if len(digits) == 10:
        return _isbn10_to_13(digits)
    return digits


def _isbn10_checksum(d: str) -> bool:
    if not re.fullmatch(r"[0-9]{9}[0-9X]", d):
        return False
    total = sum((10 - i) * (10 if c == "X" else int(c)) for i, c in enumerate(d))
    return total % 11 == 0


def _isbn13_checksum(d: str) -> bool:
    if not d.isdigit():
        return False
    total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(d))
    return total % 10 == 0


def _isbn10_to_13(isbn10: str) -> str:
    core = "978" + isbn10[:9]
    total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(core))
    check = (10 - total % 10) % 10
    return core + str(check)


def normalize_author_key(family: str, given: str | None = None) -> str:
    """'Martinez-Garcia', 'Luis' -> 'martinez-garcia,l'.

    Lowercase, strip accents, collapse whitespace to dashes inside the family
    name. First-initial only from the given name."""
    if not family:
        return ","
    fam = _strip_accents(family).strip().lower()
    fam = re.sub(r"\s+", "-", fam)
    fam = re.sub(r"[^a-z0-9\-]", "", fam)
    initial = ""
    if given:
        first = _strip_accents(given).strip()
        if first:
            initial = first[0].lower()
    return f"{fam},{initial}"


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
