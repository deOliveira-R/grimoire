"""MCP search ``section`` filter: only chunks tagged with the requested
section surface as snippets. Item ranking is unaffected."""

from __future__ import annotations

import sqlite3

import pytest

from grimoire.mcp import tools


@pytest.fixture
def seeded_sections(tmp_db: sqlite3.Connection) -> sqlite3.Connection:
    """One paper with three chunks, each in a different section. The shared
    query word ``boron`` hits every chunk so selection is driven by the
    section filter, not by keyword-relevance."""
    cur = tmp_db.execute(
        "INSERT INTO items(item_type, title, abstract, metadata_source, metadata_confidence) "
        "VALUES ('paper', 'Boron studies', 'A boron study.', 'manual', 1.0)"
    )
    item_id = int(cur.lastrowid)  # type: ignore[arg-type]
    for idx, (section, text) in enumerate(
        [
            ("introduction", "boron behavior has been studied for decades"),
            ("methods", "we dissolved boron in distilled water at various temperatures"),
            ("results", "boron concentration climbed linearly with exposure"),
        ]
    ):
        tmp_db.execute(
            "INSERT INTO chunks(item_id, chunk_index, page, text, section) "
            "VALUES (?, ?, ?, ?, ?)",
            (item_id, idx, None, text, section),
        )
    return tmp_db


def test_section_filter_narrows_snippet_to_methods(
    seeded_sections: sqlite3.Connection,
) -> None:
    hits = tools.search(
        seeded_sections, "boron", mode="keyword", section="methods", limit=5
    )
    assert hits
    main = hits[0]
    assert main.snippet is not None
    assert "dissolved" in main.snippet.text


def test_section_filter_narrows_snippet_to_results(
    seeded_sections: sqlite3.Connection,
) -> None:
    hits = tools.search(
        seeded_sections, "boron", mode="keyword", section="results", limit=5
    )
    assert hits
    main = hits[0]
    assert main.snippet is not None
    assert "climbed linearly" in main.snippet.text


def test_unknown_section_yields_no_snippet(
    seeded_sections: sqlite3.Connection,
) -> None:
    # No chunk is tagged 'conclusion' → item still ranked, but snippet empty.
    hits = tools.search(
        seeded_sections, "boron", mode="keyword", section="conclusion", limit=5
    )
    assert hits
    assert hits[0].snippet is None


def test_no_section_returns_any_snippet(
    seeded_sections: sqlite3.Connection,
) -> None:
    hits = tools.search(seeded_sections, "boron", mode="keyword", limit=5)
    assert hits
    assert hits[0].snippet is not None
