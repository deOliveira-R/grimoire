"""Parse GROBID TEI XML into a compact Claude-friendly dict.

Input: the ``grobid_tei`` artifact bytes as produced by
``extract.grobid.extract_fulltext``.

Output shape (all keys optional except ``sections`` and ``references``):

    {
      "header": {"title", "abstract", "doi", "year", "authors": [...]},
      "sections": [
         {"level": 1, "heading": "Introduction", "text": "..."},
         {"level": 1, "heading": "Methods",      "text": "..."},
         ...
      ],
      "references": [
         {"authors": [...], "title": "...", "year": 2024, "doi": "10...", "venue": "...", "raw": "..."},
         ...
      ]
    }

We keep it deliberately simple — TEI lets you go deep (affiliations,
coordinates, equations), but the MCP consumer wants flat, fast-to-scan JSON.
Anyone who needs the lossless form pulls the artifact bytes directly."""

from __future__ import annotations

import logging
from typing import Any
from xml.etree import ElementTree as ET

log = logging.getLogger(__name__)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
_TEI_TAG = "{http://www.tei-c.org/ns/1.0}"


def parse_structure(tei_bytes: bytes) -> dict[str, Any] | None:
    try:
        root = ET.fromstring(tei_bytes)
    except ET.ParseError as exc:
        log.warning("invalid TEI: %s", exc)
        return None

    return {
        "header": _header(root),
        "sections": _sections(root),
        "references": _references(root),
    }


# ---------- header ----------------------------------------------------------


def _header(root: ET.Element) -> dict[str, Any]:
    h = root.find(".//tei:teiHeader", TEI_NS)
    if h is None:
        return {}

    title = _text_of(h.find(".//tei:fileDesc/tei:titleStmt/tei:title", TEI_NS))
    abstract_node = h.find(".//tei:profileDesc/tei:abstract", TEI_NS)
    abstract = None
    if abstract_node is not None:
        parts = [
            _text_of(p)
            for p in abstract_node.findall(".//tei:p", TEI_NS) or [abstract_node]
        ]
        abstract = " ".join(p for p in parts if p) or None

    doi = None
    for idno in h.findall(".//tei:idno", TEI_NS):
        if (idno.attrib.get("type") or "").lower() == "doi" and idno.text:
            doi = idno.text.strip()
            break

    year = None
    for date in h.findall(".//tei:imprint/tei:date", TEI_NS):
        raw = date.attrib.get("when") or (date.text or "")
        for tok in (raw,):
            if len(tok) >= 4 and tok[:4].isdigit():
                year = int(tok[:4])
                break
        if year:
            break

    return {
        "title": title,
        "abstract": abstract,
        "doi": doi,
        "year": year,
        "authors": _authors_in(h, analytic_only=True),
    }


def _authors_in(scope: ET.Element, *, analytic_only: bool) -> list[dict[str, str | None]]:
    # In TEI, <analytic> holds *this paper's* authors; <monogr> can hold the
    # host journal/book — analytic_only=True keeps us in the right scope.
    base_xpath = (
        ".//tei:sourceDesc//tei:analytic/tei:author/tei:persName"
        if analytic_only
        else ".//tei:author/tei:persName"
    )
    out: list[dict[str, str | None]] = []
    for pers in scope.findall(base_xpath, TEI_NS):
        family = _text_of(pers.find("tei:surname", TEI_NS))
        given = " ".join(
            e.text.strip() for e in pers.findall("tei:forename", TEI_NS) if e.text
        ).strip() or None
        if family:
            out.append({"family": family, "given": given})
    return out


# ---------- body sections --------------------------------------------------


def _sections(root: ET.Element) -> list[dict[str, Any]]:
    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is None:
        return []
    sections: list[dict[str, Any]] = []
    for div in body.findall("tei:div", TEI_NS):
        heading_el = div.find("tei:head", TEI_NS)
        heading = _text_of(heading_el) or ""
        level = _section_level(heading_el)
        # Concatenate all paragraphs in this top-level div. Nested sub-divs
        # keep their paragraphs; we flatten for the MCP consumer.
        paras: list[str] = []
        for p in div.findall(".//tei:p", TEI_NS):
            t = _text_of(p)
            if t:
                paras.append(t)
        text = "\n\n".join(paras).strip()
        if heading or text:
            sections.append(
                {
                    "level": level,
                    "heading": heading,
                    "text": text,
                }
            )
    return sections


def _section_level(head: ET.Element | None) -> int:
    """Best-effort: TEI doesn't always annotate depth. Top-level divs = 1;
    nested = derived from ancestor div count."""
    if head is None:
        return 1
    # Walk up to count <div> ancestors — xml.etree doesn't expose parents so
    # we use the 'n' attribute when present (GROBID emits n='1', n='1.2', …).
    n = head.attrib.get("n") or ""
    if n:
        return n.count(".") + 1
    return 1


# ---------- references -----------------------------------------------------


def _references(root: ET.Element) -> list[dict[str, Any]]:
    """Parse the bibliography out of ``<back><div><listBibl>``."""
    bibs = root.findall(".//tei:back//tei:listBibl/tei:biblStruct", TEI_NS)
    out: list[dict[str, Any]] = []
    for b in bibs:
        # GROBID emits ``<analytic><title/>`` even when it can't extract an
        # article title (common for older journal-article references where
        # only the venue + volume + page is in the bibliography). Fall back
        # to ``monogr/title`` whenever the analytic title resolves to empty,
        # not just when the element is absent.
        title = _text_of(b.find("tei:analytic/tei:title", TEI_NS))
        if title is None:
            title = _text_of(b.find("tei:monogr/tei:title", TEI_NS))
        authors = _authors_in(b, analytic_only=False)
        year = None
        for date in b.findall(".//tei:imprint/tei:date", TEI_NS):
            when = date.attrib.get("when") or (date.text or "")
            if len(when) >= 4 and when[:4].isdigit():
                year = int(when[:4])
                break
        doi = None
        for idno in b.findall(".//tei:idno", TEI_NS):
            if (idno.attrib.get("type") or "").lower() == "doi" and idno.text:
                doi = idno.text.strip()
                break
        venue = _text_of(b.find(".//tei:monogr/tei:title", TEI_NS))
        raw = _text_of(b.find(".//tei:note", TEI_NS)) or None
        out.append(
            {
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "venue": venue,
                "raw": raw,
            }
        )
    return out


# ---------- helpers --------------------------------------------------------


def _text_of(el: ET.Element | None) -> str | None:
    if el is None:
        return None
    text = " ".join("".join(el.itertext()).split()).strip()
    return text or None
