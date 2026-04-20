"""GROBID header extractor. Produces a Metadata with title/abstract/authors/DOI/year
parsed directly from the PDF — so we're no longer hostage to Crossref's ~99%
abstract-miss rate (see Phase 2 oracle findings).

Fails soft: if the service is unreachable, returns None and the caller falls
back to the Crossref/arXiv path."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from grimoire.config import settings
from grimoire.models import Author, Metadata

log = logging.getLogger(__name__)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def extract_header(path: Path) -> Metadata | None:
    if not settings.grobid_url:
        return None
    url = settings.grobid_url.rstrip("/") + "/api/processHeaderDocument"
    try:
        with path.open("rb") as fh:
            r = httpx.post(
                url,
                files={"input": (path.name, fh, "application/pdf")},
                # consolidateHeader=0 → no external lookups; we'll corroborate via Crossref ourselves
                data={"consolidateHeader": "0"},
                timeout=60.0,
            )
        r.raise_for_status()
    except Exception as exc:
        log.warning("GROBID header extract failed for %s: %s", path, exc)
        return None
    return parse_tei(r.text)


def parse_tei(tei_xml: str) -> Metadata | None:
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as exc:
        log.warning("GROBID returned invalid TEI: %s", exc)
        return None

    header = root.find(".//tei:teiHeader", TEI_NS)
    if header is None:
        return None

    title = _first_text(header, ".//tei:fileDesc/tei:titleStmt/tei:title")
    abstract = _abstract(header)
    authors = _authors(header)
    doi = _doi(header)
    year = _year(header)
    venue = _first_text(header, ".//tei:monogr/tei:title")

    # If we got nothing useful, return None so the caller treats it as a miss.
    if not (title or abstract or authors or doi):
        return None

    return Metadata(
        title=title,
        abstract=abstract,
        publication_year=year,
        doi=doi,
        venue=venue,
        authors=authors,
        source="grobid",
        confidence=0.85,
        raw={"grobid_tei": tei_xml},
    )


def _first_text(root: ET.Element, xpath: str) -> str | None:
    el = root.find(xpath, TEI_NS)
    if el is None:
        return None
    text = "".join(el.itertext()).strip()
    return text or None


def _abstract(header: ET.Element) -> str | None:
    node = header.find(".//tei:profileDesc/tei:abstract", TEI_NS)
    if node is None:
        return None
    parts = []
    for p in node.findall(".//tei:p", TEI_NS) or [node]:
        text = "".join(p.itertext()).strip()
        if text:
            parts.append(text)
    joined = " ".join(parts)
    return joined or None


def _authors(header: ET.Element) -> list[Author]:
    out = []
    for pers in header.findall(".//tei:sourceDesc//tei:analytic/tei:author/tei:persName", TEI_NS):
        fam = _first_text(pers, "tei:surname")
        given_parts = [e.text for e in pers.findall("tei:forename", TEI_NS) if e.text]
        given = " ".join(given_parts).strip() or None
        if fam:
            out.append(Author(family_name=fam, given_name=given))
    return out


_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s]+)\b")


def _doi(header: ET.Element) -> str | None:
    # Explicit idno element first
    for idno in header.findall(".//tei:idno", TEI_NS):
        itype = (idno.attrib.get("type") or "").lower()
        if itype == "doi" and idno.text:
            return idno.text.strip()
    # Fallback: any 10.x/... pattern in header text
    text = " ".join(header.itertext())
    m = _DOI_RE.search(text)
    return m.group(1).rstrip(".,;)") if m else None


def _year(header: ET.Element) -> int | None:
    for date in header.findall(".//tei:imprint/tei:date", TEI_NS):
        when = date.attrib.get("when") or (date.text or "")
        m = re.search(r"(1[89]\d{2}|20\d{2}|21\d{2})", when)
        if m:
            return int(m.group(1))
    return None


def ping(url: str | None = None, timeout: float = 3.0) -> bool:
    """Cheap reachability check for the GROBID service."""
    target = (url or settings.grobid_url or "").rstrip("/")
    if not target:
        return False
    try:
        r = httpx.get(f"{target}/api/isalive", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _extract_xml_for_tests(raw: str) -> dict[str, Any]:
    """Small helper exposed for tests: return a dict of parsed fields without
    constructing a Metadata."""
    md = parse_tei(raw)
    if md is None:
        return {}
    return {
        "title": md.title,
        "abstract": md.abstract,
        "year": md.publication_year,
        "doi": md.doi,
        "venue": md.venue,
        "authors": [(a.family_name, a.given_name) for a in md.authors],
    }
