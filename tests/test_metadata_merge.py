"""Tests for merge_metadata_layered — the multi-resolver fusion used at ingest time."""

from __future__ import annotations

import pytest

from grimoire.models import Author, Metadata, merge_metadata_layered


def test_crossref_plus_grobid_keeps_crossref_bibliographic_fills_abstract() -> None:
    crossref = Metadata(
        title="A paper",
        doi="10.1/x",
        venue="J. Stuff",
        volume="10",
        issue="2",
        pages="1-10",
        publication_year=2024,
        authors=[Author(family_name="Alice"), Author(family_name="Bob")],
        source="crossref",
        confidence=1.0,
        raw={"crossref": {"k": 1}},
    )
    grobid = Metadata(
        title="A paper (parsed)",
        abstract="The abstract from the PDF.",
        doi="10.1/x",
        authors=[Author(family_name="Alice"), Author(family_name="Bob")],
        source="grobid",
        confidence=0.85,
        raw={"grobid_tei": "..."},
    )

    merged = merge_metadata_layered([grobid, crossref])
    # Crossref wins overall (higher authority) → bibliographic fields preserved
    assert merged.source == "crossref"
    assert merged.venue == "J. Stuff"
    assert merged.volume == "10"
    assert merged.pages == "1-10"
    # Abstract was None in Crossref, filled from GROBID
    assert merged.abstract == "The abstract from the PDF."
    # Raw blobs unioned
    assert merged.raw is not None
    assert "crossref" in merged.raw and "grobid_tei" in merged.raw


def test_grobid_alone_is_kept_when_crossref_missing() -> None:
    grobid = Metadata(
        title="A paper",
        abstract="An abstract.",
        doi="10.1/x",
        authors=[Author(family_name="Alice")],
        source="grobid",
        confidence=0.85,
    )
    merged = merge_metadata_layered([grobid])
    assert merged.source == "grobid"
    assert merged.abstract == "An abstract."


def test_authority_order_crossref_over_arxiv_over_grobid() -> None:
    cr = Metadata(title="CR", source="crossref", confidence=1.0)
    ax = Metadata(title="AX", source="arxiv", confidence=1.0)
    gb = Metadata(title="GB", source="grobid", confidence=0.85)
    merged = merge_metadata_layered([gb, ax, cr])
    assert merged.title == "CR"


def test_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        merge_metadata_layered([])


def test_authors_fallback_when_primary_empty() -> None:
    cr = Metadata(title="X", source="crossref", confidence=1.0, authors=[])
    gb = Metadata(
        title="X",
        source="grobid",
        confidence=0.85,
        authors=[Author(family_name="Alice"), Author(family_name="Bob")],
    )
    merged = merge_metadata_layered([cr, gb])
    assert [a.family_name for a in merged.authors] == ["Alice", "Bob"]


def test_primary_authors_not_overwritten() -> None:
    cr = Metadata(
        title="X",
        source="crossref",
        confidence=1.0,
        authors=[Author(family_name="CrossrefAuthor")],
    )
    gb = Metadata(
        title="X",
        source="grobid",
        confidence=0.85,
        authors=[Author(family_name="GrobidAuthor")],
    )
    merged = merge_metadata_layered([gb, cr])
    assert [a.family_name for a in merged.authors] == ["CrossrefAuthor"]
