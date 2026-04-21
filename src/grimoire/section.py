"""Classify a TEI section heading into a coarse section type.

Output values come from a small fixed vocabulary the MCP consumer can reason
about; anything unfamiliar maps to ``'other'``. Matching is substring-based
so compound headings like ``"Materials and Methods"`` still land in the right
bucket."""

from __future__ import annotations

import re

SectionType = str

# Ordered pairs: (needle, section). First match wins, so put more-specific
# needles before less-specific ones when they would otherwise collide.
_HEADING_MAP: list[tuple[str, SectionType]] = [
    ("introduction", "introduction"),
    ("background", "introduction"),
    ("related work", "introduction"),
    ("method", "methods"),
    ("methodolog", "methods"),
    ("experimental", "methods"),
    ("materials", "methods"),
    ("result", "results"),
    ("finding", "results"),
    ("discussion", "discussion"),
    ("conclusion", "conclusion"),
    ("summary", "conclusion"),
]

# Strip leading section numbering like "1.", "2.3", "III.", "IV  ".
_NUMBER_PREFIX = re.compile(r"^([IVXLCDM]+\.?\s+|\d+(?:\.\d+)*\.?\s+)", re.IGNORECASE)


def classify(heading: str | None) -> SectionType:
    if not heading:
        return "other"
    h = _NUMBER_PREFIX.sub("", heading.lower().strip())
    for needle, section in _HEADING_MAP:
        if needle in h:
            return section
    return "other"
