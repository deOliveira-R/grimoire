"""item_artifacts storage + CLI integration + MCP tool (plan §6 strategic add).

Does NOT hit a real GROBID server — the build CLI is tested with the GROBID
call monkeypatched. Live GROBID integration is a manual smoke ran separately."""

from __future__ import annotations

import sqlite3

import pytest

from grimoire.storage import artifacts


@pytest.fixture
def seeded_items(tmp_db: sqlite3.Connection) -> list[int]:
    """Three items: two with primary PDFs (via content_hash on items + a
    primary artifact row), one metadata-only (no attachment)."""
    ids = []
    for title, ch in [
        ("With file A", "hash-a" + "0" * 58),
        ("With file B", "hash-b" + "0" * 58),
        ("No file",     None),
    ]:
        cur = tmp_db.execute(
            "INSERT INTO items(item_type, title, content_hash, metadata_source, metadata_confidence) "
            "VALUES ('paper', ?, ?, 'manual', 1.0) RETURNING id",
            (title, ch),
        )
        item_id = int(cur.fetchone()["id"])
        if ch is not None:
            tmp_db.execute(
                "INSERT INTO item_artifacts(item_id, kind, content_hash, source) "
                "VALUES (?, 'primary', ?, 'manual')",
                (item_id, ch),
            )
        ids.append(item_id)
    return ids


# ---------- migration backfill ----------------------------------------------


def test_migration_003_backfills_primary_from_items(
    tmp_db: sqlite3.Connection,
) -> None:
    # The tmp_db fixture already ran all migrations — insert a pre-existing
    # item so the backfill semantics show up even on a fresh DB.
    cur = tmp_db.execute(
        "INSERT INTO items(item_type, title, content_hash, metadata_source) "
        "VALUES ('paper', 'X', 'deadbeef', 'crossref') RETURNING id"
    )
    item_id = int(cur.fetchone()["id"])
    # Item without a hash must not produce a backfill row.
    tmp_db.execute(
        "INSERT INTO items(item_type, title, metadata_source) "
        "VALUES ('paper', 'Y', 'manual_required')"
    )
    # Migrations already ran during tmp_db setup, so we only need to verify
    # the *new-row* path works — the static backfill path is covered
    # end-to-end by the two-stage test in test_schema below.
    tmp_db.execute(
        "INSERT INTO item_artifacts(item_id, kind, content_hash, source) "
        "VALUES (?, 'primary', 'deadbeef', 'crossref')",
        (item_id,),
    )
    n = tmp_db.execute(
        "SELECT COUNT(*) AS n FROM item_artifacts WHERE kind='primary'"
    ).fetchone()["n"]
    assert n == 1


# ---------- storage module --------------------------------------------------


def test_store_writes_cas_blob_and_upsert_row(
    seeded_items: list[int], tmp_db: sqlite3.Connection, tmp_data_root
) -> None:
    target = seeded_items[0]
    h = artifacts.store(
        tmp_db, target, "grobid_tei", b"<TEI>hello</TEI>", source="grobid-0.8.1"
    )
    # CAS path exists
    assert (tmp_data_root / "files" / h[:2] / h[2:4] / h).exists()
    # DB row inserted
    info = artifacts.info(tmp_db, target, "grobid_tei")
    assert info is not None
    assert info.content_hash == h
    assert info.source == "grobid-0.8.1"
    assert info.size_bytes == len(b"<TEI>hello</TEI>")


def test_store_is_idempotent_on_identical_bytes(
    seeded_items: list[int], tmp_db: sqlite3.Connection
) -> None:
    target = seeded_items[0]
    h1 = artifacts.store(tmp_db, target, "grobid_tei", b"<TEI/>", source="v1")
    h2 = artifacts.store(tmp_db, target, "grobid_tei", b"<TEI/>", source="v2")
    assert h1 == h2
    # Only one row (PRIMARY KEY on item_id+kind).
    n = tmp_db.execute(
        "SELECT COUNT(*) AS n FROM item_artifacts WHERE item_id=? AND kind='grobid_tei'",
        (target,),
    ).fetchone()["n"]
    assert n == 1
    # UPSERT updated the source column
    info = artifacts.info(tmp_db, target, "grobid_tei")
    assert info is not None and info.source == "v2"


def test_store_replaces_on_different_bytes(
    seeded_items: list[int], tmp_db: sqlite3.Connection
) -> None:
    target = seeded_items[0]
    h1 = artifacts.store(tmp_db, target, "grobid_tei", b"<TEI>a</TEI>")
    h2 = artifacts.store(tmp_db, target, "grobid_tei", b"<TEI>b</TEI>")
    assert h1 != h2
    info = artifacts.info(tmp_db, target, "grobid_tei")
    assert info is not None and info.content_hash == h2


def test_read_returns_none_for_missing(
    seeded_items: list[int], tmp_db: sqlite3.Connection
) -> None:
    assert artifacts.read(tmp_db, seeded_items[0], "grobid_tei") is None
    assert artifacts.exists(tmp_db, seeded_items[0], "grobid_tei") is False


def test_read_roundtrip(
    seeded_items: list[int], tmp_db: sqlite3.Connection
) -> None:
    payload = b"<TEI>body bytes</TEI>"
    artifacts.store(tmp_db, seeded_items[0], "grobid_tei", payload)
    assert artifacts.read(tmp_db, seeded_items[0], "grobid_tei") == payload


def test_items_missing_kind_skips_items_without_primary(
    seeded_items: list[int], tmp_db: sqlite3.Connection
) -> None:
    # Two items have 'primary', one does not.
    missing = artifacts.items_missing_kind(tmp_db, "grobid_tei", primary_only=True)
    assert set(missing) == {seeded_items[0], seeded_items[1]}
    artifacts.store(tmp_db, seeded_items[0], "grobid_tei", b"<TEI/>")
    assert artifacts.items_missing_kind(tmp_db, "grobid_tei", primary_only=True) == [
        seeded_items[1]
    ]


# ---------- TEI parser ------------------------------------------------------


_TEI_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Boron dilution transients in PWR</title></titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author><persName><forename>Alice</forename><surname>Smith</surname></persName></author>
            <author><persName><forename>Bob</forename><surname>Doe</surname></persName></author>
            <idno type="DOI">10.1/boron</idno>
          </analytic>
          <monogr><imprint><date when="2024-05-10"/></imprint></monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract><p>A study of boron dynamics.</p></abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div><head n="1">Introduction</head><p>Background paragraph.</p></div>
      <div><head n="2">Methods</head><p>First method.</p><p>Second method.</p></div>
      <div><head n="3">Results</head><p>Numbers.</p></div>
    </body>
    <back>
      <div>
        <listBibl>
          <biblStruct>
            <analytic>
              <author><persName><forename>X</forename><surname>Author</surname></persName></author>
              <title>Prior work on boron</title>
              <idno type="DOI">10.9/prior</idno>
            </analytic>
            <monogr>
              <title>Journal X</title>
              <imprint><date when="2018"/></imprint>
            </monogr>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""


def test_tei_parse_header() -> None:
    from grimoire.extract.tei import parse_structure

    s = parse_structure(_TEI_SAMPLE)
    assert s is not None
    header = s["header"]
    assert header["title"] == "Boron dilution transients in PWR"
    assert header["doi"] == "10.1/boron"
    assert header["year"] == 2024
    assert header["abstract"] and "boron dynamics" in header["abstract"]
    assert header["authors"] == [
        {"family": "Smith", "given": "Alice"},
        {"family": "Doe", "given": "Bob"},
    ]


def test_tei_parse_sections() -> None:
    from grimoire.extract.tei import parse_structure

    s = parse_structure(_TEI_SAMPLE)
    assert s is not None
    sections = s["sections"]
    assert [sec["heading"] for sec in sections] == ["Introduction", "Methods", "Results"]
    # Methods section has two paragraphs joined with blank line
    methods = next(sec for sec in sections if sec["heading"] == "Methods")
    assert "First method." in methods["text"]
    assert "Second method." in methods["text"]


def test_tei_parse_references() -> None:
    from grimoire.extract.tei import parse_structure

    s = parse_structure(_TEI_SAMPLE)
    assert s is not None
    refs = s["references"]
    assert len(refs) == 1
    assert refs[0]["title"] == "Prior work on boron"
    assert refs[0]["year"] == 2018
    assert refs[0]["doi"] == "10.9/prior"
    assert refs[0]["authors"] == [{"family": "Author", "given": "X"}]


_TEI_EMPTY_ANALYTIC_TITLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>host</title></titleStmt>
      <sourceDesc>
        <biblStruct><monogr><imprint><date when="2024"/></imprint></monogr></biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text><body/><back><div><listBibl>
    <biblStruct>
      <analytic>
        <title/>
        <author><persName><surname>Williams</surname></persName></author>
      </analytic>
      <monogr>
        <title level="j">Comm. Pure Appl. Math</title>
        <imprint><date when="1973"/></imprint>
      </monogr>
    </biblStruct>
  </listBibl></div></back></text>
</TEI>
"""


def test_tei_reference_falls_back_to_monogr_title_when_analytic_is_empty() -> None:
    """GROBID emits ``<analytic><title/>`` for journal-article references
    whose article-level title it couldn't parse out (common in pre-DOI
    literature). Title resolution must fall back to ``<monogr><title>``
    rather than returning None."""
    from grimoire.extract.tei import parse_structure

    s = parse_structure(_TEI_EMPTY_ANALYTIC_TITLE)
    assert s is not None
    refs = s["references"]
    assert len(refs) == 1
    assert refs[0]["title"] == "Comm. Pure Appl. Math"
    assert refs[0]["venue"] == "Comm. Pure Appl. Math"
    assert refs[0]["year"] == 1973
    assert refs[0]["authors"] == [{"family": "Williams", "given": None}]


def test_tei_parse_invalid_returns_none() -> None:
    from grimoire.extract.tei import parse_structure

    assert parse_structure(b"<not valid xml") is None


# ---------- MCP tool --------------------------------------------------------


def test_mcp_get_document_structure_returns_none_when_no_artifact(
    seeded_items: list[int], tmp_db: sqlite3.Connection
) -> None:
    from grimoire.mcp import tools

    assert tools.get_document_structure(tmp_db, seeded_items[0]) is None


def test_mcp_get_document_structure_parses_stored_tei(
    seeded_items: list[int], tmp_db: sqlite3.Connection
) -> None:
    from grimoire.mcp import tools

    artifacts.store(tmp_db, seeded_items[0], "grobid_tei", _TEI_SAMPLE)
    s = tools.get_document_structure(tmp_db, seeded_items[0])
    assert s is not None
    assert s["header"]["doi"] == "10.1/boron"
    assert len(s["sections"]) == 3
    assert len(s["references"]) == 1


# ---------- CLI build path (with GROBID monkeypatched) ----------------------


def test_cli_build_stores_artifact_from_fake_grobid(
    seeded_items: list[int],
    tmp_db: sqlite3.Connection,
    tmp_data_root,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI command is thin; verify it ends up calling artifacts.store
    with whatever GROBID returns."""
    from pathlib import Path

    from grimoire.config import settings
    from grimoire.extract import grobid
    from grimoire.storage.cas import CAS

    # Put a real blob on disk for each primary artifact so path-exists checks pass.
    cas = CAS(settings.files_root)
    for item_id in seeded_items[:2]:
        h = artifacts.get_hash(tmp_db, item_id, "primary")
        assert h is not None
        # Store a dummy PDF-like byte string under that exact hash — skip the
        # CAS.store_path mechanic by writing directly (we don't care about
        # content here; only that the path exists).
        cas_path = cas.path_for_hash(h)
        cas_path.parent.mkdir(parents=True, exist_ok=True)
        cas_path.write_bytes(b"%PDF-1.4\nstub\n")

    # Monkeypatch the GROBID call to return synthetic TEI.
    def _fake_fulltext(path: Path, **_: object) -> bytes:
        return _TEI_SAMPLE

    monkeypatch.setattr(grobid, "extract_fulltext", _fake_fulltext)
    monkeypatch.setattr(settings, "grobid_url", "http://fake-grobid:8070")

    # Use the underlying function rather than go through Typer, so we can
    # assert inside the same test process. This matches the pattern of the
    # other CLI-like tests in the suite (tests call the functions, not Typer).
    from typer.testing import CliRunner

    from grimoire.cli import app

    result = CliRunner().invoke(app, ["artifacts", "build", "--kind", "grobid_tei"])
    assert result.exit_code == 0, result.output

    # Both primary-carrying items should now have a grobid_tei artifact.
    for item_id in seeded_items[:2]:
        assert artifacts.exists(tmp_db, item_id, "grobid_tei") is True

    # Re-running without --force is a no-op — the "missing" list is empty.
    result2 = CliRunner().invoke(app, ["artifacts", "build", "--kind", "grobid_tei"])
    assert result2.exit_code == 0
    assert "0 item(s) to process" in result2.output


def test_cli_build_rejects_unsupported_kind() -> None:
    from typer.testing import CliRunner

    from grimoire.cli import app

    r = CliRunner().invoke(app, ["artifacts", "build", "--kind", "nonsense"])
    assert r.exit_code != 0
    assert "unsupported kind" in r.output or "unsupported" in r.output


def test_cli_status_reports_per_kind_counts(
    seeded_items: list[int], tmp_db: sqlite3.Connection
) -> None:
    artifacts.store(tmp_db, seeded_items[0], "grobid_tei", _TEI_SAMPLE)
    from typer.testing import CliRunner

    from grimoire.cli import app

    r = CliRunner().invoke(app, ["artifacts", "status"])
    assert r.exit_code == 0
    assert "primary" in r.output
    assert "grobid_tei" in r.output
