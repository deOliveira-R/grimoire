"""Resolver unit tests. HTTP is monkeypatched — no real network calls.
Real-network regression tests would be marked @pytest.mark.network and run on demand."""

from __future__ import annotations

from typing import Any

import pytest

from grimoire.models import Author, Metadata
from grimoire.resolve import arxiv_api, crossref, openlibrary


def test_crossref_doi_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = {
        "message": {
            "DOI": "10.1234/example",
            "title": ["A Paper"],
            "abstract": "<p>The abstract.</p>",
            "issued": {"date-parts": [[2024, 3, 15]]},
            "container-title": ["Journal of Examples"],
            "volume": "12",
            "issue": "3",
            "page": "1-10",
            "author": [
                {
                    "family": "Smith",
                    "given": "John",
                    "ORCID": "http://orcid.org/0000-0001-0000-0001",
                },
                {"family": "Doe", "given": "Jane"},
            ],
        }
    }

    monkeypatch.setattr(crossref, "_fetch_raw", lambda doi: sample)

    md = crossref.resolve("10.1234/example")
    assert md.doi == "10.1234/example"
    assert md.title == "A Paper"
    assert md.publication_year == 2024
    assert md.venue == "Journal of Examples"
    assert md.volume == "12" and md.issue == "3" and md.pages == "1-10"
    assert md.source == "crossref"
    assert md.confidence == 1.0
    assert md.authors == [
        Author(family_name="Smith", given_name="John", orcid="0000-0001-0000-0001"),
        Author(family_name="Doe", given_name="Jane", orcid=None),
    ]
    assert "<" not in (md.abstract or "")


def test_crossref_missing_doi_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crossref, "_fetch_raw", lambda doi: None)
    assert crossref.resolve("10.9999/missing") is None


def test_arxiv_id_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {
        "entry_id": "http://arxiv.org/abs/2401.01234v1",
        "title": "An Arxiv Paper",
        "summary": "Short abstract.",
        "published_year": 2024,
        "authors": [{"name": "Alice Researcher"}, {"name": "Bob Scientist"}],
        "doi": "10.1111/arxpublished",
        "journal_ref": "Journal of Things, 2024",
    }
    monkeypatch.setattr(arxiv_api, "_fetch_raw", lambda arxiv_id: fake)

    md = arxiv_api.resolve("2401.01234")
    assert md.arxiv_id == "2401.01234"
    assert md.title == "An Arxiv Paper"
    assert md.abstract == "Short abstract."
    assert md.publication_year == 2024
    assert md.source == "arxiv"
    assert md.confidence == 1.0
    # linked DOI is captured via raw so ingest can link preprint → published
    assert md.raw and md.raw.get("linked_doi") == "10.1111/arxpublished"
    assert [a.family_name for a in md.authors] == ["Researcher", "Scientist"]


def test_openlibrary_isbn_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    sample: dict[str, Any] = {
        "title": "Sample Book",
        "publish_date": "2019",
        "publishers": [{"name": "Acme Press"}],
        "authors": [{"name": "Alice Writer"}, {"name": "Bob Writer"}],
        "identifiers": {"isbn_13": ["9780131101630"]},
    }
    monkeypatch.setattr(openlibrary, "_fetch_raw", lambda isbn: sample)

    md = openlibrary.resolve("9780131101630")
    assert md.isbn == "9780131101630"
    assert md.title == "Sample Book"
    assert md.publication_year == 2019
    assert md.venue == "Acme Press"
    assert md.source == "openlibrary"
    assert md.confidence == 0.9
    assert [a.family_name for a in md.authors] == ["Writer", "Writer"]


def test_openlibrary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openlibrary, "_fetch_raw", lambda isbn: None)
    assert openlibrary.resolve("9999999999999") is None


def test_metadata_merge_prefers_higher_source(monkeypatch: pytest.MonkeyPatch) -> None:
    # Crossref > arXiv > OpenLibrary > LLM (plan §3 merge semantics).
    from grimoire.models import prefer_more_authoritative

    cr = Metadata(title="From CR", source="crossref", confidence=1.0)
    ax = Metadata(title="From AX", source="arxiv", confidence=1.0)
    chosen = prefer_more_authoritative([ax, cr])
    assert chosen.title == "From CR"
