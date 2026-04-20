"""End-to-end migration tests using a synthetic Zotero SQLite.

Zotero's real schema is large; we seed only the tables the migrator
actually reads. Each test builds the minimum fixture for the scenario
under test."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from grimoire.migrate.zotero import migrate


# ---------- synthetic Zotero SQLite builder --------------------------------


_ZOTERO_SCHEMA = """
CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
CREATE TABLE items (
    itemID INTEGER PRIMARY KEY,
    itemTypeID INTEGER REFERENCES itemTypes(itemTypeID),
    key TEXT
);
CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT UNIQUE);
CREATE TABLE itemData (
    itemID INTEGER,
    fieldID INTEGER,
    valueID INTEGER,
    PRIMARY KEY(itemID, fieldID)
);
CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE creators (
    creatorID INTEGER PRIMARY KEY,
    lastName TEXT,
    firstName TEXT
);
CREATE TABLE creatorTypes (
    creatorTypeID INTEGER PRIMARY KEY,
    creatorType TEXT UNIQUE
);
CREATE TABLE itemCreators (
    itemID INTEGER,
    creatorID INTEGER,
    creatorTypeID INTEGER,
    orderIndex INTEGER,
    PRIMARY KEY(itemID, creatorID, creatorTypeID)
);
CREATE TABLE tags (tagID INTEGER PRIMARY KEY, name TEXT UNIQUE);
CREATE TABLE itemTags (
    itemID INTEGER,
    tagID INTEGER,
    PRIMARY KEY(itemID, tagID)
);
CREATE TABLE collections (
    collectionID INTEGER PRIMARY KEY,
    collectionName TEXT,
    parentCollectionID INTEGER
);
CREATE TABLE collectionItems (
    itemID INTEGER,
    collectionID INTEGER,
    PRIMARY KEY(itemID, collectionID)
);
CREATE TABLE itemAttachments (
    itemID INTEGER PRIMARY KEY,
    parentItemID INTEGER,
    contentType TEXT,
    path TEXT
);
"""


_CREATOR_TYPES = ["author", "editor", "translator", "bookAuthor", "contributor", "commenter"]
_FIELDS = [
    "title",
    "date",
    "DOI",
    "publicationTitle",
    "volume",
    "issue",
    "pages",
    "series",
    "seriesNumber",
    "edition",
    "abstractNote",
    "ISBN",
    "language",
    "extra",
]


class ZoteroBuilder:
    """Tiny DSL for seeding Zotero-shaped rows into a blank SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_ZOTERO_SCHEMA)
        for i, ct in enumerate(_CREATOR_TYPES, start=1):
            self.conn.execute(
                "INSERT INTO creatorTypes(creatorTypeID, creatorType) VALUES (?,?)", (i, ct)
            )
        for i, fn in enumerate(_FIELDS, start=1):
            self.conn.execute(
                "INSERT INTO fields(fieldID, fieldName) VALUES (?,?)", (i, fn)
            )
        self._next_value_id = 1
        self._next_creator_id = 1

    def item_type(self, type_id: int, name: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO itemTypes(itemTypeID, typeName) VALUES (?, ?)",
            (type_id, name),
        )

    def item(self, item_id: int, type_name: str, key: str = "") -> None:
        tid = self.conn.execute(
            "SELECT itemTypeID FROM itemTypes WHERE typeName=?", (type_name,)
        ).fetchone()
        if tid is None:
            next_tid = (
                self.conn.execute("SELECT COALESCE(MAX(itemTypeID),0)+1 FROM itemTypes").fetchone()[
                    0
                ]
            )
            self.item_type(next_tid, type_name)
            tid = (next_tid,)
        self.conn.execute(
            "INSERT INTO items(itemID, itemTypeID, key) VALUES (?, ?, ?)",
            (item_id, tid[0], key),
        )

    def field(self, item_id: int, name: str, value: str) -> None:
        fid = self.conn.execute(
            "SELECT fieldID FROM fields WHERE fieldName=?", (name,)
        ).fetchone()[0]
        vid = self._next_value_id
        self._next_value_id += 1
        self.conn.execute(
            "INSERT INTO itemDataValues(valueID, value) VALUES (?, ?)", (vid, value)
        )
        self.conn.execute(
            "INSERT INTO itemData(itemID, fieldID, valueID) VALUES (?, ?, ?)",
            (item_id, fid, vid),
        )

    def creator(
        self, item_id: int, last: str, first: str | None, role: str, order: int
    ) -> None:
        cid = self._next_creator_id
        self._next_creator_id += 1
        self.conn.execute(
            "INSERT INTO creators(creatorID, lastName, firstName) VALUES (?, ?, ?)",
            (cid, last, first),
        )
        ctid = self.conn.execute(
            "SELECT creatorTypeID FROM creatorTypes WHERE creatorType=?", (role,)
        ).fetchone()[0]
        self.conn.execute(
            "INSERT INTO itemCreators(itemID, creatorID, creatorTypeID, orderIndex) "
            "VALUES (?, ?, ?, ?)",
            (item_id, cid, ctid, order),
        )

    def tag(self, item_id: int, name: str) -> None:
        tag_id = self.conn.execute(
            "INSERT OR IGNORE INTO tags(name) VALUES (?) RETURNING tagID", (name,)
        ).fetchone()
        if tag_id is None:
            tag_id = self.conn.execute(
                "SELECT tagID FROM tags WHERE name=?", (name,)
            ).fetchone()
        self.conn.execute(
            "INSERT INTO itemTags(itemID, tagID) VALUES (?, ?)", (item_id, tag_id[0])
        )

    def collection(
        self, collection_id: int, name: str, parent: int | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO collections(collectionID, collectionName, parentCollectionID) "
            "VALUES (?, ?, ?)",
            (collection_id, name, parent),
        )

    def in_collection(self, item_id: int, collection_id: int) -> None:
        self.conn.execute(
            "INSERT INTO collectionItems(itemID, collectionID) VALUES (?, ?)",
            (item_id, collection_id),
        )

    def attachment(
        self,
        attachment_item_id: int,
        parent_item_id: int,
        zotero_key: str,
        filename: str,
        content_type: str = "application/pdf",
    ) -> None:
        """Register a PDF attachment. Callers must still place the file on
        disk at ``storage/{key}/{filename}`` for it to be found."""
        self.item(attachment_item_id, "attachment", key=zotero_key)
        self.conn.execute(
            "INSERT INTO itemAttachments(itemID, parentItemID, contentType, path) "
            "VALUES (?, ?, ?, ?)",
            (attachment_item_id, parent_item_id, content_type, f"storage:{filename}"),
        )

    def commit(self) -> None:
        self.conn.commit()
        self.conn.close()


@pytest.fixture
def zotero_env(tmp_path: Path) -> tuple[Path, Path]:
    """Return (zotero_db_path, zotero_storage_dir)."""
    db = tmp_path / "zotero.sqlite"
    storage = tmp_path / "storage"
    storage.mkdir()
    return db, storage


# ---------- tests -----------------------------------------------------------


def test_imports_journal_article_with_authors_and_fields(
    zotero_env: tuple[Path, Path], tmp_db: sqlite3.Connection, tmp_data_root: Path
) -> None:
    db, storage = zotero_env
    b = ZoteroBuilder(db)
    b.item(1, "journalArticle")
    b.field(1, "title", "Boron dilution in PWR")
    b.field(1, "date", "2024-05-10")
    b.field(1, "DOI", "10.1/boron")
    b.field(1, "publicationTitle", "Nucl. Eng. Design")
    b.field(1, "volume", "450")
    b.field(1, "pages", "111-120")
    b.field(1, "abstractNote", "A study of boron dynamics.")
    b.field(1, "language", "en")
    b.creator(1, "Smith", "Alice", "author", 0)
    b.creator(1, "Doe", "Bob", "author", 1)
    b.tag(1, "safety")
    b.commit()

    report = migrate(tmp_db, library_path=db, storage_dir=storage)
    assert report.inserted == 1
    assert report.pdf_attachments_stored == 0  # no attachment in this test

    row = tmp_db.execute(
        """SELECT item_type, title, publication_year, doi, venue, volume, pages,
                  abstract, language, metadata_source
           FROM items WHERE doi = '10.1/boron'"""
    ).fetchone()
    assert row is not None
    assert row["item_type"] == "paper"
    assert row["title"] == "Boron dilution in PWR"
    assert row["publication_year"] == 2024
    assert row["venue"] == "Nucl. Eng. Design"
    assert row["volume"] == "450"
    assert row["pages"] == "111-120"
    assert row["abstract"] == "A study of boron dynamics."
    assert row["language"] == "en"
    assert row["metadata_source"] == "zotero_import"

    authors = tmp_db.execute(
        """SELECT a.family_name, a.given_name, ia.role
           FROM item_authors ia JOIN authors a ON a.id = ia.author_id
           WHERE ia.item_id = ? ORDER BY ia.position""",
        (row[0] if isinstance(row, tuple) else _item_id_by_doi(tmp_db, "10.1/boron"),),
    ).fetchall()
    # Depending on row tuple/dict semantics, fetch authors explicitly
    item_id = _item_id_by_doi(tmp_db, "10.1/boron")
    authors = tmp_db.execute(
        """SELECT a.family_name, ia.role
           FROM item_authors ia JOIN authors a ON a.id = ia.author_id
           WHERE ia.item_id = ? ORDER BY ia.position""",
        (item_id,),
    ).fetchall()
    names = [(a["family_name"], a["role"]) for a in authors]
    assert names == [("Smith", "author"), ("Doe", "author")]

    tags = [
        r["name"]
        for r in tmp_db.execute(
            """SELECT t.name FROM item_tags it JOIN tags t ON t.id = it.tag_id
               WHERE it.item_id = ?""",
            (item_id,),
        ).fetchall()
    ]
    assert tags == ["safety"]


def _item_id_by_doi(conn: sqlite3.Connection, doi: str) -> int:
    return int(conn.execute("SELECT id FROM items WHERE doi = ?", (doi,)).fetchone()["id"])


def test_book_with_editors_keeps_editor_role(
    zotero_env: tuple[Path, Path], tmp_db: sqlite3.Connection, tmp_data_root: Path
) -> None:
    db, storage = zotero_env
    b = ZoteroBuilder(db)
    b.item(1, "book")
    b.field(1, "title", "Handbook of reactor physics")
    b.field(1, "ISBN", "978-1234567890")
    b.field(1, "edition", "2")
    b.field(1, "date", "2020")
    b.creator(1, "Alpha", "A", "editor", 0)
    b.creator(1, "Beta", "B", "editor", 1)
    b.creator(1, "Gamma", "G", "author", 2)  # chapter author attached to book
    b.commit()

    report = migrate(tmp_db, library_path=db, storage_dir=storage)
    assert report.inserted == 1

    item_id = tmp_db.execute(
        "SELECT id FROM items WHERE isbn = '978-1234567890'"
    ).fetchone()["id"]

    roles = tmp_db.execute(
        """SELECT a.family_name, ia.role FROM item_authors ia
           JOIN authors a ON a.id = ia.author_id WHERE ia.item_id = ?
           ORDER BY ia.position""",
        (item_id,),
    ).fetchall()
    assert [(r["family_name"], r["role"]) for r in roles] == [
        ("Alpha", "editor"),
        ("Beta", "editor"),
        ("Gamma", "author"),
    ]
    edition_row = tmp_db.execute(
        "SELECT edition FROM items WHERE id = ?", (item_id,)
    ).fetchone()
    assert edition_row["edition"] == "2"


def test_deleted_items_skipped(
    zotero_env: tuple[Path, Path], tmp_db: sqlite3.Connection, tmp_data_root: Path
) -> None:
    db, storage = zotero_env
    b = ZoteroBuilder(db)
    b.item(1, "journalArticle")
    b.field(1, "title", "Kept")
    b.field(1, "DOI", "10.1/kept")
    b.item(2, "journalArticle")
    b.field(2, "title", "Dropped")
    b.field(2, "DOI", "10.1/dropped")
    b.conn.execute("INSERT INTO deletedItems(itemID) VALUES (2)")
    b.commit()

    report = migrate(tmp_db, library_path=db, storage_dir=storage)
    assert report.inserted == 1
    titles = {
        r["title"] for r in tmp_db.execute("SELECT title FROM items").fetchall()
    }
    assert titles == {"Kept"}


def test_idempotent_rerun(
    zotero_env: tuple[Path, Path], tmp_db: sqlite3.Connection, tmp_data_root: Path
) -> None:
    db, storage = zotero_env
    b = ZoteroBuilder(db)
    b.item(1, "journalArticle")
    b.field(1, "title", "X")
    b.field(1, "DOI", "10.1/x")
    b.commit()

    r1 = migrate(tmp_db, library_path=db, storage_dir=storage)
    assert r1.inserted == 1
    r2 = migrate(tmp_db, library_path=db, storage_dir=storage)
    assert r2.inserted == 0
    assert r2.skipped_already_imported == 1
    # Only one grimoire item exists
    assert (
        tmp_db.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"] == 1
    )


def test_dry_run_does_not_insert(
    zotero_env: tuple[Path, Path], tmp_db: sqlite3.Connection, tmp_data_root: Path
) -> None:
    db, storage = zotero_env
    b = ZoteroBuilder(db)
    b.item(1, "journalArticle")
    b.field(1, "title", "Not imported")
    b.field(1, "DOI", "10.1/z")
    b.commit()

    report = migrate(tmp_db, library_path=db, storage_dir=storage, dry_run=True)
    assert report.total_candidates == 1
    assert report.inserted == 0
    assert tmp_db.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"] == 0


def test_pdf_attachment_is_stored_and_linked(
    zotero_env: tuple[Path, Path], tmp_db: sqlite3.Connection, tmp_data_root: Path
) -> None:
    db, storage = zotero_env
    pdf_key = "AAAABBBB"
    pdf_filename = "paper.pdf"
    (storage / pdf_key).mkdir()
    (storage / pdf_key / pdf_filename).write_bytes(b"%PDF-1.4\nfake pdf bytes\n")

    bld = ZoteroBuilder(db)
    bld.item(1, "journalArticle")
    bld.field(1, "title", "X")
    bld.field(1, "DOI", "10.1/x")
    bld.creator(1, "Smith", "A", "author", 0)
    bld.attachment(2, 1, pdf_key, pdf_filename)
    bld.commit()

    report = migrate(tmp_db, library_path=db, storage_dir=storage)
    assert report.inserted == 1
    assert report.pdf_attachments_stored == 1

    row = tmp_db.execute(
        "SELECT content_hash, file_path FROM items WHERE doi = '10.1/x'"
    ).fetchone()
    assert row["content_hash"] is not None
    assert row["file_path"].endswith(row["content_hash"])
    assert (tmp_data_root / "files" / row["file_path"]).exists()


def test_collections_create_parent_child_hierarchy(
    zotero_env: tuple[Path, Path], tmp_db: sqlite3.Connection, tmp_data_root: Path
) -> None:
    db, storage = zotero_env
    b = ZoteroBuilder(db)
    b.collection(10, "Nuclear", parent=None)
    b.collection(11, "Reactor safety", parent=10)
    b.item(1, "journalArticle")
    b.field(1, "title", "Nested coll item")
    b.field(1, "DOI", "10.1/nest")
    b.in_collection(1, 11)
    b.in_collection(1, 10)
    b.commit()

    migrate(tmp_db, library_path=db, storage_dir=storage)

    cols = tmp_db.execute(
        "SELECT id, name, parent_id FROM collections ORDER BY name"
    ).fetchall()
    # Both collections were created
    names = {c["name"] for c in cols}
    assert names == {"Nuclear", "Reactor safety"}
    # Child points at parent when parent was seen first
    child = next(c for c in cols if c["name"] == "Reactor safety")
    parent = next(c for c in cols if c["name"] == "Nuclear")
    assert child["parent_id"] == parent["id"]


def test_minor_creator_roles_dropped(
    zotero_env: tuple[Path, Path], tmp_db: sqlite3.Connection, tmp_data_root: Path
) -> None:
    db, storage = zotero_env
    b = ZoteroBuilder(db)
    b.item(1, "journalArticle")
    b.field(1, "title", "Minor roles")
    b.field(1, "DOI", "10.1/minor")
    b.creator(1, "Main", "A", "author", 0)
    b.creator(1, "CommenterPerson", "C", "commenter", 1)
    b.commit()

    migrate(tmp_db, library_path=db, storage_dir=storage)
    item_id = _item_id_by_doi(tmp_db, "10.1/minor")
    authors = tmp_db.execute(
        "SELECT a.family_name FROM item_authors ia JOIN authors a ON a.id = ia.author_id "
        "WHERE ia.item_id = ? ORDER BY ia.position",
        (item_id,),
    ).fetchall()
    assert [a["family_name"] for a in authors] == ["Main"]


def test_attachment_conflicts_merge_with_existing_item(
    zotero_env: tuple[Path, Path], tmp_db: sqlite3.Connection, tmp_data_root: Path
) -> None:
    """Zotero stores two entries for the same DOI (preprint + published);
    migration runs tier-1 dedup and merges into the first-imported grimoire row."""
    db, storage = zotero_env
    b = ZoteroBuilder(db)
    b.item(1, "journalArticle")
    b.field(1, "title", "Shared paper")
    b.field(1, "DOI", "10.9/shared")
    b.item(2, "journalArticle")
    b.field(2, "title", "Shared paper (alt)")
    b.field(2, "DOI", "10.9/shared")
    b.commit()

    report = migrate(tmp_db, library_path=db, storage_dir=storage)
    assert report.inserted == 1
    assert report.merged == 1
    assert tmp_db.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"] == 1
