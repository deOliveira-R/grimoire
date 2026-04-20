"""OPDS router tests — parse every feed as XML, walk it, and assert the
entries/links we promise.

The router uses ``grimoire.db.connect()`` without a path, so the
``tmp_data_root`` fixture (which monkeypatches ``settings.data_root``) wires
the per-request connection to the test database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from grimoire.app import app
from grimoire.storage.cas import CAS

ATOM_NS = "http://www.w3.org/2005/Atom"
OPDS_NS = "http://opds-spec.org/2010/catalog"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
DC_NS = "http://purl.org/dc/terms/"


@pytest.fixture
def seeded_db(tmp_db: sqlite3.Connection, tmp_data_root: Path) -> sqlite3.Connection:
    """Library with:
      - paper p1 (authors: Smith, Doe; tag 'safety'; collection 'Reactor safety';
        real file in CAS under a stable hash)
      - preprint p2 (author: Smith)
      - book b1 (editor: Editor)"""
    cas = CAS(tmp_data_root / "files")
    # A byte string stable across test runs so the content hash is known.
    pdf_bytes = b"%PDF-1.4\n% test blob for opds download test\n"
    content_hash, _ = cas.store(pdf_bytes)

    def add_item(**cols: object) -> int:
        cols.setdefault("metadata_source", "crossref")
        cols.setdefault("metadata_confidence", 1.0)
        columns = ",".join(cols.keys())
        placeholders = ",".join("?" * len(cols))
        cur = tmp_db.execute(
            f"INSERT INTO items({columns}) VALUES ({placeholders})",
            tuple(cols.values()),
        )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def add_author(item_id: int, family: str, given: str | None, pos: int) -> None:
        key = f"{family.lower()},{(given or '')[:1].lower()}"
        tmp_db.execute(
            "INSERT OR IGNORE INTO authors(family_name, given_name, normalized_key) VALUES (?,?,?)",
            (family, given, key),
        )
        aid = tmp_db.execute(
            "SELECT id FROM authors WHERE normalized_key=? AND orcid IS NULL", (key,)
        ).fetchone()["id"]
        tmp_db.execute(
            "INSERT OR IGNORE INTO item_authors(item_id, author_id, position, role) "
            "VALUES (?,?,?, 'author')",
            (item_id, aid, pos),
        )

    p1 = add_item(
        item_type="paper",
        title="Boron dilution transients in PWR",
        abstract="A study of boron concentration during accident conditions.",
        publication_year=2024,
        doi="10.1/boron",
        venue="Nucl. Eng. Design",
        language="en",
        content_hash=content_hash,
        file_path="/tmp/boron.pdf",
    )
    add_author(p1, "Smith", "Alice", 0)
    add_author(p1, "Doe", "Bob", 1)

    p2 = add_item(
        item_type="preprint",
        title="Boron dilution transients in PWR (preprint)",
        publication_year=2023,
        arxiv_id="2301.11111",
    )
    add_author(p2, "Smith", "Alice", 0)

    b1 = add_item(
        item_type="book",
        title="Nuclear reactor physics handbook",
        publication_year=2020,
        isbn="978-0-12-345678-9",
    )
    add_author(b1, "Editor", "Em", 0)

    tmp_db.execute("INSERT INTO tags(name) VALUES ('safety')")
    tmp_db.execute(
        "INSERT INTO item_tags(item_id, tag_id) SELECT ?, id FROM tags WHERE name='safety'",
        (p1,),
    )
    tmp_db.execute("INSERT INTO collections(name) VALUES ('Reactor safety')")
    tmp_db.execute(
        "INSERT INTO item_collections(item_id, collection_id) "
        "SELECT ?, id FROM collections WHERE name='Reactor safety'",
        (p1,),
    )
    return tmp_db


@pytest.fixture
def client(seeded_db: sqlite3.Connection) -> TestClient:
    return TestClient(app)


def parse(body: bytes) -> ET.Element:
    return ET.fromstring(body)


def entry_titles(root: ET.Element) -> list[str]:
    return [e.findtext(f"{{{ATOM_NS}}}title", default="") for e in root.findall(f"{{{ATOM_NS}}}entry")]


def links_by_rel(elem: ET.Element, rel: str) -> list[ET.Element]:
    return [link for link in elem.findall(f"{{{ATOM_NS}}}link") if link.get("rel") == rel]


# ---------- root nav ---------------------------------------------------------


def test_root_feed_returns_opds_navigation(client: TestClient) -> None:
    r = client.get("/opds")
    assert r.status_code == 200
    assert "application/atom+xml" in r.headers["content-type"]
    assert "profile=opds-catalog" in r.headers["content-type"]
    assert "kind=navigation" in r.headers["content-type"]
    root = parse(r.content)
    assert root.tag == f"{{{ATOM_NS}}}feed"
    titles = entry_titles(root)
    assert "Recent additions" in titles
    assert "Collections" in titles
    assert "Tags" in titles
    assert "Authors" in titles
    assert "By item type" in titles
    # Self link + search descriptor link
    self_links = links_by_rel(root, "self")
    assert self_links and self_links[0].get("href") == "/opds"
    search_links = links_by_rel(root, "search")
    assert search_links
    assert search_links[0].get("href") == "/opds/opensearch.xml"


def test_root_trailing_slash_also_works(client: TestClient) -> None:
    r = client.get("/opds/")
    assert r.status_code == 200


# ---------- acquisition feeds ------------------------------------------------


def test_recent_feed_lists_all_items_as_acquisition(client: TestClient) -> None:
    r = client.get("/opds/recent")
    assert r.status_code == 200
    assert "kind=acquisition" in r.headers["content-type"]
    root = parse(r.content)
    titles = entry_titles(root)
    assert "Boron dilution transients in PWR" in titles
    assert "Boron dilution transients in PWR (preprint)" in titles
    assert "Nuclear reactor physics handbook" in titles


def test_recent_entry_has_download_link_for_cas_item(client: TestClient) -> None:
    r = client.get("/opds/recent")
    root = parse(r.content)
    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        if entry.findtext(f"{{{ATOM_NS}}}title") == "Boron dilution transients in PWR":
            acq_links = [
                link
                for link in entry.findall(f"{{{ATOM_NS}}}link")
                if link.get("rel") == "http://opds-spec.org/acquisition"
            ]
            assert acq_links, "p1 has a content_hash, must carry an acquisition link"
            href = acq_links[0].get("href") or ""
            assert href.startswith("/files/")
            assert acq_links[0].get("type") == "application/pdf"
            # dc:identifier uses the DOI when present
            dcid = entry.findtext(f"{{{DC_NS}}}identifier")
            assert dcid == "urn:doi:10.1/boron"
            return
    pytest.fail("p1 entry not found")


def test_recent_entry_without_file_has_no_acquisition_link(client: TestClient) -> None:
    r = client.get("/opds/recent")
    root = parse(r.content)
    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        if "preprint" in entry.findtext(f"{{{ATOM_NS}}}title", default="").lower():
            acq = [
                link
                for link in entry.findall(f"{{{ATOM_NS}}}link")
                if link.get("rel") == "http://opds-spec.org/acquisition"
            ]
            assert not acq, "preprint without a hash must not expose a download link"
            return
    pytest.fail("preprint entry not found")


def test_recent_pagination(client: TestClient) -> None:
    # Page size 2, 3 items → expect next link on page 1, prev link on page 2.
    r = client.get("/opds/recent", params={"limit": 2, "offset": 0})
    root = parse(r.content)
    assert len(root.findall(f"{{{ATOM_NS}}}entry")) == 2
    assert links_by_rel(root, "next"), "should have a next link"
    assert not links_by_rel(root, "prev"), "page 1 has no prev"

    r2 = client.get("/opds/recent", params={"limit": 2, "offset": 2})
    root2 = parse(r2.content)
    assert len(root2.findall(f"{{{ATOM_NS}}}entry")) == 1
    assert links_by_rel(root2, "prev")
    assert not links_by_rel(root2, "next")


# ---------- collections ------------------------------------------------------


def test_collections_nav_lists_only_non_empty(client: TestClient) -> None:
    r = client.get("/opds/collections")
    assert r.status_code == 200
    root = parse(r.content)
    titles = entry_titles(root)
    assert titles == ["Reactor safety"]


def test_collection_acquisition(client: TestClient) -> None:
    # There's exactly one collection in the seed — grab its id dynamically.
    r_nav = client.get("/opds/collections")
    root = parse(r_nav.content)
    subsection = links_by_rel(root.find(f"{{{ATOM_NS}}}entry"), "subsection")
    href = subsection[0].get("href")
    assert href is not None
    r = client.get(href)
    assert r.status_code == 200
    root = parse(r.content)
    assert entry_titles(root) == ["Boron dilution transients in PWR"]


def test_unknown_collection_returns_404(client: TestClient) -> None:
    assert client.get("/opds/collections/99999").status_code == 404


# ---------- tags -------------------------------------------------------------


def test_tags_nav(client: TestClient) -> None:
    r = client.get("/opds/tags")
    assert r.status_code == 200
    root = parse(r.content)
    assert entry_titles(root) == ["safety"]


def test_tag_acquisition(client: TestClient) -> None:
    r = client.get("/opds/tags/safety")
    assert r.status_code == 200
    root = parse(r.content)
    assert entry_titles(root) == ["Boron dilution transients in PWR"]


def test_unknown_tag_returns_404(client: TestClient) -> None:
    assert client.get("/opds/tags/nonexistent").status_code == 404


# ---------- authors ----------------------------------------------------------


def test_authors_nav(client: TestClient) -> None:
    r = client.get("/opds/authors")
    assert r.status_code == 200
    root = parse(r.content)
    titles = entry_titles(root)
    # Three distinct authors across three items
    assert "Alice Smith" in titles
    assert "Bob Doe" in titles
    assert "Em Editor" in titles


def test_author_items(client: TestClient) -> None:
    r_nav = client.get("/opds/authors")
    nav = parse(r_nav.content)
    # Find Alice Smith's subsection
    smith_href = None
    for entry in nav.findall(f"{{{ATOM_NS}}}entry"):
        if entry.findtext(f"{{{ATOM_NS}}}title") == "Alice Smith":
            smith_href = links_by_rel(entry, "subsection")[0].get("href")
            break
    assert smith_href is not None
    r = client.get(smith_href)
    assert r.status_code == 200
    root = parse(r.content)
    titles = entry_titles(root)
    # Alice is author on both paper and preprint
    assert len(titles) == 2
    assert any("Boron dilution transients in PWR" == t for t in titles)


# ---------- item types -------------------------------------------------------


def test_types_nav_lists_only_non_empty(client: TestClient) -> None:
    r = client.get("/opds/types")
    assert r.status_code == 200
    titles = entry_titles(parse(r.content))
    assert set(titles) == {"Paper", "Preprint", "Book"}


def test_type_filter(client: TestClient) -> None:
    r = client.get("/opds/types/book")
    assert r.status_code == 200
    root = parse(r.content)
    assert entry_titles(root) == ["Nuclear reactor physics handbook"]


def test_unknown_type_returns_404(client: TestClient) -> None:
    assert client.get("/opds/types/not-a-thing").status_code == 404


# ---------- search -----------------------------------------------------------


def test_opensearch_descriptor(client: TestClient) -> None:
    r = client.get("/opds/opensearch.xml")
    assert r.status_code == 200
    root = parse(r.content)
    assert root.tag == f"{{{OPENSEARCH_NS}}}OpenSearchDescription"
    template = root.find(f"{{{OPENSEARCH_NS}}}Url").get("template")
    assert "{searchTerms}" in template


def test_search_finds_item_by_title(client: TestClient) -> None:
    r = client.get("/opds/search", params={"q": "boron"})
    assert r.status_code == 200
    root = parse(r.content)
    titles = entry_titles(root)
    assert any("Boron dilution" in t for t in titles)
    # OpenSearch metadata present
    total = root.findtext(f"{{{OPENSEARCH_NS}}}totalResults")
    assert total is not None and int(total) >= 1


def test_search_safe_from_fts_operators(client: TestClient) -> None:
    # Would crash FTS5 parsing if the query weren't phrase-quoted.
    r = client.get("/opds/search", params={"q": 'NOT ) boron OR'})
    assert r.status_code == 200


def test_search_requires_nonempty_query(client: TestClient) -> None:
    assert client.get("/opds/search").status_code == 422


# ---------- venues ----------------------------------------------------------


def test_venues_nav_lists_distinct_journals(client: TestClient) -> None:
    r = client.get("/opds/venues")
    assert r.status_code == 200
    titles = entry_titles(parse(r.content))
    # Only paper p1 has a venue in the seed fixture
    assert "Nucl. Eng. Design" in titles


def test_venue_acquisition(client: TestClient) -> None:
    from urllib.parse import quote

    r = client.get(f"/opds/venues/{quote('Nucl. Eng. Design', safe='')}")
    assert r.status_code == 200
    root = parse(r.content)
    assert entry_titles(root) == ["Boron dilution transients in PWR"]


def test_unknown_venue_returns_404(client: TestClient) -> None:
    assert client.get("/opds/venues/NoSuchJournal").status_code == 404


# ---------- years -----------------------------------------------------------


def test_years_nav_lists_distinct_years(client: TestClient) -> None:
    r = client.get("/opds/years")
    assert r.status_code == 200
    titles = entry_titles(parse(r.content))
    # Seed items have years 2024 (paper), 2023 (preprint), 2020 (book)
    assert set(titles) == {"2024", "2023", "2020"}


def test_year_acquisition(client: TestClient) -> None:
    r = client.get("/opds/years/2024")
    assert r.status_code == 200
    assert entry_titles(parse(r.content)) == ["Boron dilution transients in PWR"]


def test_unknown_year_returns_404(client: TestClient) -> None:
    assert client.get("/opds/years/1900").status_code == 404


def test_root_feed_advertises_journals_and_years(client: TestClient) -> None:
    r = client.get("/opds")
    titles = entry_titles(parse(r.content))
    assert "Journals" in titles
    assert "Years" in titles
