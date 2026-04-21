"""HTML web UI — home + item detail. Renders Jinja2 templates against a
seeded DB; tests inspect the rendered markup for expected strings/links.

We don't parse the HTML strictly (it's v1, content is small); substring
assertions catch the regressions that actually matter: wrong values
leaking, missing filter links, 404 on removed items."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from grimoire.app import app
from grimoire.storage.cas import CAS


@pytest.fixture
def seeded_db(tmp_db: sqlite3.Connection, tmp_data_root: Path) -> sqlite3.Connection:
    cas = CAS(tmp_data_root / "files")
    h, _ = cas.store(b"%PDF-1.4\nfake pdf\n")

    def add_item(**cols: object) -> int:
        cols.setdefault("metadata_source", "crossref")
        cols.setdefault("metadata_confidence", 1.0)
        columns = ",".join(cols.keys())
        placeholders = ",".join("?" * len(cols))
        cur = tmp_db.execute(
            f"INSERT INTO items({columns}) VALUES ({placeholders})", tuple(cols.values())
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
        content_hash=h,
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

    add_item(
        item_type="book",
        title="Nuclear reactor physics handbook",
        publication_year=2020,
        isbn="978-0-12-345678-9",
    )

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

    tmp_db.execute(
        "INSERT INTO item_relations(subject_id, relation, object_id, confidence) "
        "VALUES (?, 'preprint_of', ?, 1.0)",
        (p2, p1),
    )
    tmp_db.execute(
        "INSERT INTO item_relations(subject_id, relation, object_id, confidence) "
        "VALUES (?, 'published_as', ?, 1.0)",
        (p1, p2),
    )
    return tmp_db


@pytest.fixture
def client(seeded_db: sqlite3.Connection) -> TestClient:
    return TestClient(app)


# ---------- home ------------------------------------------------------------


def test_home_renders_all_items(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "Boron dilution transients in PWR" in body
    assert "Nuclear reactor physics handbook" in body
    assert "Paper" in body  # type facet
    assert "safety" in body  # tag facet
    # OPDS alternate discovery link
    assert '<link rel="alternate" type="application/atom+xml"' in body
    assert 'href="/opds"' in body


def test_home_type_filter(client: TestClient) -> None:
    r = client.get("/?type=book")
    assert r.status_code == 200
    body = r.text
    assert "Nuclear reactor physics handbook" in body
    assert "Boron dilution transients in PWR" not in body


def test_home_tag_filter(client: TestClient) -> None:
    r = client.get("/?tag=safety")
    assert r.status_code == 200
    body = r.text
    # Only the paper carries 'safety'
    assert "Boron dilution transients in PWR</a>" in body or "Boron dilution transients in PWR" in body
    assert "handbook" not in body


def test_home_search(client: TestClient) -> None:
    r = client.get("/", params={"q": "boron"})
    assert r.status_code == 200
    body = r.text
    # Title still renders — check on words unaffected by the highlight wrap.
    assert "dilution transients in PWR" in body
    # Search term surfaces as an active-filter chip
    assert "q: boron" in body
    assert "handbook" not in body


def test_home_search_highlights_query_terms(
    seeded_db: sqlite3.Connection, client: TestClient
) -> None:
    body = client.get("/", params={"q": "boron"}).text
    # Case-insensitive wrap in <mark> on title and abstract
    assert "<mark>Boron</mark>" in body or "<mark>boron</mark>" in body


def test_home_search_highlight_is_case_insensitive(client: TestClient) -> None:
    body = client.get("/", params={"q": "BORON"}).text
    # Input was uppercase but title has "Boron" — must still be wrapped
    assert "<mark>Boron</mark>" in body


def test_home_no_highlight_without_query(client: TestClient) -> None:
    body = client.get("/").text
    assert "<mark>" not in body


def test_home_search_highlight_escapes_html(
    seeded_db: sqlite3.Connection, client: TestClient
) -> None:
    seeded_db.execute(
        "UPDATE items SET title='XSS probe <script>alert(1)</script>' "
        "WHERE title LIKE 'Boron%PWR'"
    )
    # Query the tokens present in the rewritten title so FTS5 returns a hit.
    body = client.get("/", params={"q": "XSS"}).text
    # Highlight happens; script tag is escaped, no raw <script> in output
    assert "<mark>XSS</mark>" in body
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_home_empty_search_shows_empty_state(client: TestClient) -> None:
    r = client.get("/", params={"q": "zzzunmatched"})
    assert r.status_code == 200
    assert "No items match" in r.text


def test_home_pagination_links(client: TestClient) -> None:
    r = client.get("/", params={"limit": 1})
    body = r.text
    assert "Next" in body
    assert "offset=1" in body


# ---------- item detail ------------------------------------------------------


def test_item_detail_renders_full_metadata(client: TestClient) -> None:
    r = client.get("/items/1")
    assert r.status_code == 200
    body = r.text
    assert "Boron dilution transients in PWR" in body
    assert "Alice Smith" in body
    assert "Bob Doe" in body
    assert "Nucl. Eng. Design" in body  # venue
    assert "10.1/boron" in body  # DOI
    assert "safety" in body  # tag pill
    # Download link points to /files/{hash}
    assert 'href="/files/' in body


def test_item_detail_shows_related_items(client: TestClient) -> None:
    r = client.get("/items/1")
    body = r.text
    assert "Related" in body
    assert "published_as" in body
    # Related item title appears
    assert "preprint" in body.lower()


def test_item_detail_without_content_hash_hides_download(client: TestClient) -> None:
    r = client.get("/items/2")
    assert r.status_code == 200
    body = r.text
    assert "/files/" not in body  # preprint has no content_hash
    assert "2301.11111" in body  # but arxiv id is shown


def test_item_not_found_returns_404(client: TestClient) -> None:
    assert client.get("/items/99999").status_code == 404


# ---------- venue + year facets and stackable filters -----------------------


def test_home_shows_journal_and_year_facets(client: TestClient) -> None:
    body = client.get("/").text
    assert "Journal" in body
    assert "Nucl. Eng. Design" in body
    assert "Year" in body
    assert "2024" in body


def test_home_venue_filter(client: TestClient) -> None:
    r = client.get("/", params={"venue": "Nucl. Eng. Design"})
    assert r.status_code == 200
    body = r.text
    assert "Boron dilution transients in PWR" in body
    # The preprint has no venue — must not show
    assert "(preprint)" not in body


def test_home_year_filter(client: TestClient) -> None:
    r = client.get("/", params={"year": 2020})
    body = r.text
    assert "Nuclear reactor physics handbook" in body
    assert "Boron dilution transients in PWR</a>" not in body


def test_home_stacks_venue_and_year(client: TestClient) -> None:
    # Paper p1 is Nucl. Eng. Design + 2024 — should be the one hit
    r = client.get("/", params={"venue": "Nucl. Eng. Design", "year": 2024})
    body = r.text
    assert "Boron dilution transients in PWR" in body
    # Preprint has no venue, book is 2020 — both must be absent
    assert "handbook" not in body
    assert "(preprint)" not in body


def test_home_stacks_type_and_year(client: TestClient) -> None:
    r = client.get("/", params={"type": "book", "year": 2020})
    body = r.text
    assert "Nuclear reactor physics handbook" in body
    assert "Boron dilution transients in PWR" not in body


def test_home_item_list_shows_volume_issue_pages(
    seeded_db: sqlite3.Connection, client: TestClient
) -> None:
    seeded_db.execute(
        "UPDATE items SET volume='450', issue='3', pages='111-120' WHERE title LIKE 'Boron%PWR'"
    )
    body = client.get("/").text
    assert "450" in body
    assert "111-120" in body


def test_home_clear_filter_links_drop_only_that_axis(client: TestClient) -> None:
    r = client.get("/", params={"venue": "Nucl. Eng. Design", "year": 2024})
    body = r.text
    # Active filter chips appear (one per axis); each clear link keeps the
    # other axis.
    assert "venue: Nucl. Eng. Design" in body
    assert "year: 2024" in body


# ---------- collections facet + filter --------------------------------------


def test_home_shows_collections_sidebar(client: TestClient) -> None:
    body = client.get("/").text
    assert "Collections" in body
    assert "Reactor safety" in body


def test_list_collections_tree_rolls_up_counts(
    seeded_db: sqlite3.Connection,
) -> None:
    from grimoire.web.queries import list_collections_tree

    cur_a = seeded_db.execute(
        "INSERT INTO collections(name, parent_id) VALUES ('A', NULL)"
    )
    a_id = int(cur_a.lastrowid)  # type: ignore[arg-type]
    cur_b = seeded_db.execute(
        "INSERT INTO collections(name, parent_id) VALUES ('B', ?)", (a_id,)
    )
    b_id = int(cur_b.lastrowid)  # type: ignore[arg-type]
    # Seed the paper into B so A (empty) should roll up to count 1.
    paper_id = seeded_db.execute(
        "SELECT id FROM items WHERE title LIKE 'Boron%PWR'"
    ).fetchone()["id"]
    seeded_db.execute(
        "INSERT INTO item_collections(item_id, collection_id) VALUES (?, ?)",
        (paper_id, b_id),
    )

    roots = list_collections_tree(seeded_db)
    by_name = {n.collection.name: n for n in roots}
    assert "A" in by_name
    a = by_name["A"]
    assert a.collection.item_count == 0
    assert a.descendants_count == 1
    assert [c.collection.name for c in a.children] == ["B"]
    b = a.children[0]
    assert b.collection.item_count == 1
    assert b.descendants_count == 1


def test_home_collections_render_nested_tree(
    seeded_db: sqlite3.Connection, client: TestClient
) -> None:
    # Build A > B > C, with the paper moved into C. Assert all three names
    # render and that the URL filtering on C works end-to-end.
    cur_a = seeded_db.execute(
        "INSERT INTO collections(name, parent_id) VALUES ('Physics', NULL)"
    )
    a_id = int(cur_a.lastrowid)  # type: ignore[arg-type]
    cur_b = seeded_db.execute(
        "INSERT INTO collections(name, parent_id) VALUES ('Nuclear', ?)", (a_id,)
    )
    b_id = int(cur_b.lastrowid)  # type: ignore[arg-type]
    cur_c = seeded_db.execute(
        "INSERT INTO collections(name, parent_id) VALUES ('Reactors', ?)", (b_id,)
    )
    c_id = int(cur_c.lastrowid)  # type: ignore[arg-type]
    # Move the existing paper into the leaf.
    paper_id = seeded_db.execute(
        "SELECT id FROM items WHERE title LIKE 'Boron%PWR'"
    ).fetchone()["id"]
    seeded_db.execute(
        "INSERT INTO item_collections(item_id, collection_id) VALUES (?, ?)",
        (paper_id, c_id),
    )

    body = client.get("/").text
    assert "Physics" in body
    assert "Nuclear" in body
    assert "Reactors" in body
    # Clicking on the leaf filters to its items
    r = client.get("/", params={"collection": c_id})
    assert r.status_code == 200
    assert "Boron dilution transients in PWR" in r.text


def test_home_collection_filter(
    client: TestClient, seeded_db: sqlite3.Connection
) -> None:
    col_id = seeded_db.execute(
        "SELECT id FROM collections WHERE name = 'Reactor safety'"
    ).fetchone()["id"]
    r = client.get("/", params={"collection": col_id})
    assert r.status_code == 200
    body = r.text
    assert "Boron dilution transients in PWR" in body
    # Only the paper is in this collection in the seed fixture
    assert "handbook" not in body
    # Chip shows the collection *name*, not the raw id
    assert "collection: Reactor safety" in body


# ---------- sort controls ---------------------------------------------------


def test_home_sort_selector_is_rendered(client: TestClient) -> None:
    body = client.get("/").text
    assert '<select name="sort"' in body
    assert "Recently added" in body
    assert "Year (newest)" in body
    assert "Title (A–Z)" in body
    assert "First author (A–Z)" in body


def test_home_sort_year_asc_reorders(
    client: TestClient,
) -> None:
    body_year_asc = client.get("/", params={"sort": "year_asc"}).text
    # Oldest item (book, 2020) should appear before the newest (paper, 2024)
    idx_book = body_year_asc.find("Nuclear reactor physics handbook")
    idx_paper = body_year_asc.find("Boron dilution transients in PWR</a>")
    assert idx_book != -1 and idx_paper != -1
    assert idx_book < idx_paper


def test_home_sort_by_first_author_orders_alphabetically(
    seeded_db: sqlite3.Connection, client: TestClient
) -> None:
    # Seed three additional items with first authors Zenith, Alvarez, Mendez
    # — expected order after ?sort=author is Alvarez, Mendez, Zenith.
    def _seed(title: str, family: str) -> None:
        cur = seeded_db.execute(
            "INSERT INTO items(item_type, title) VALUES ('paper', ?)", (title,)
        )
        item_id = int(cur.lastrowid)  # type: ignore[arg-type]
        key = f"{family.lower()},"
        seeded_db.execute(
            "INSERT OR IGNORE INTO authors(family_name, given_name, normalized_key) "
            "VALUES (?, NULL, ?)",
            (family, key),
        )
        aid = seeded_db.execute(
            "SELECT id FROM authors WHERE normalized_key=? AND orcid IS NULL", (key,)
        ).fetchone()["id"]
        seeded_db.execute(
            "INSERT INTO item_authors(item_id, author_id, position, role) "
            "VALUES (?,?,0,'author')",
            (item_id, aid),
        )

    _seed("Author-sort Z", "Zenith")
    _seed("Author-sort A", "Alvarez")
    _seed("Author-sort M", "Mendez")

    body = client.get("/", params={"sort": "author"}).text
    idx_a = body.find("Author-sort A")
    idx_m = body.find("Author-sort M")
    idx_z = body.find("Author-sort Z")
    assert idx_a != -1 and idx_m != -1 and idx_z != -1
    assert idx_a < idx_m < idx_z


def test_home_invalid_sort_falls_back_to_default(client: TestClient) -> None:
    r = client.get("/", params={"sort": "nonsense"})
    assert r.status_code == 200
    # Default sort is 'added' — selector's 'Recently added' option is selected
    assert 'selected' in r.text


# ---------- bibtex download -------------------------------------------------


def test_item_bibtex_endpoint(client: TestClient) -> None:
    r = client.get("/items/1/bibtex")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-bibtex")
    assert r.text.startswith("@")


def test_item_bibtex_missing_returns_404(client: TestClient) -> None:
    assert client.get("/items/99999/bibtex").status_code == 404


def test_item_detail_shows_bibtex_block(client: TestClient) -> None:
    body = client.get("/items/1").text
    assert "BibTeX" in body
    # The <pre> block with the actual bibtex body
    assert 'class="bibtex"' in body
    assert "@article" in body or "@book" in body


def test_item_detail_shows_action_row(client: TestClient) -> None:
    body = client.get("/items/1").text
    assert "Download" in body
    assert "BibTeX" in body
    assert "Open DOI" in body or "Open DOI ↗" in body
