"""Heading-to-section classifier."""

from __future__ import annotations

import pytest

from grimoire.section import classify


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("Introduction", "introduction"),
        ("1. Introduction", "introduction"),
        ("Background and Motivation", "introduction"),
        ("Related Work", "introduction"),
        ("2. Methods", "methods"),
        ("2.1. Methodology", "methods"),
        ("Materials and Methods", "methods"),
        ("Experimental Setup", "methods"),
        ("III. Results", "results"),
        ("Results and Discussion", "results"),
        ("Findings", "results"),
        ("4 Discussion", "discussion"),
        ("Conclusion", "conclusion"),
        ("Summary", "conclusion"),
        # Unknown headings fall through to 'other'.
        ("Acknowledgements", "other"),
        ("Méthodes", "other"),
        (None, "other"),
        ("", "other"),
    ],
)
def test_classify(heading: str | None, expected: str) -> None:
    assert classify(heading) == expected


def test_numbered_prefix_stripped() -> None:
    # Roman + arabic + multi-level all collapse to the classifier's dictionary.
    assert classify("IV. Conclusion") == "conclusion"
    assert classify("2.3.1 Results") == "results"
    assert classify("3.  Methods") == "methods"
