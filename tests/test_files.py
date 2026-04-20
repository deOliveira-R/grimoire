"""Tests for the CAS file streamer (/files/{content_hash}).

The router uses ``grimoire.db.connect()`` with no path, so it picks up the
tmp_data_root-monkeypatched DB automatically."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from grimoire.app import app
from grimoire.storage.cas import CAS


@pytest.fixture
def pdf_seeded(tmp_db: sqlite3.Connection, tmp_data_root: Path) -> tuple[str, bytes]:
    """Store a small PDF in CAS and register an item that references it.
    Returns (content_hash, raw_bytes) for test assertions."""
    cas = CAS(tmp_data_root / "files")
    blob = b"%PDF-1.4\ncas blob bytes that the download should return verbatim\n"
    h, _ = cas.store(blob)
    tmp_db.execute(
        """INSERT INTO items(item_type, title, content_hash, file_path, metadata_source, metadata_confidence)
           VALUES ('paper', 'A test paper', ?, '/tmp/paper.pdf', 'manual', 1.0)""",
        (h,),
    )
    return h, blob


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_download_returns_raw_bytes(
    client: TestClient, pdf_seeded: tuple[str, bytes]
) -> None:
    h, blob = pdf_seeded
    r = client.get(f"/files/{h}")
    assert r.status_code == 200
    assert r.content == blob
    assert r.headers["content-type"] == "application/pdf"


def test_download_sets_original_filename(
    client: TestClient, pdf_seeded: tuple[str, bytes]
) -> None:
    h, _ = pdf_seeded
    r = client.get(f"/files/{h}")
    cd = r.headers.get("content-disposition", "")
    assert 'filename="paper.pdf"' in cd or "filename=paper.pdf" in cd


def test_download_unknown_hash_returns_404(
    client: TestClient, tmp_db: sqlite3.Connection
) -> None:
    # Valid shape, no matching item
    r = client.get("/files/" + "0" * 64)
    assert r.status_code == 404


def test_download_malformed_hash_rejected(client: TestClient) -> None:
    r = client.get("/files/not-a-real-hash")
    assert r.status_code == 400


def test_download_path_traversal_rejected(client: TestClient) -> None:
    # Would be scary if this got through; the regex should reject.
    r = client.get("/files/../../../etc/passwd")
    # FastAPI returns 404 for unmatched path first (the slash breaks the pattern)
    assert r.status_code in (400, 404)


def test_download_hash_without_file_on_disk_returns_404(
    tmp_db: sqlite3.Connection, tmp_data_root: Path, client: TestClient
) -> None:
    # Item registered, CAS path empty — DB says yes, disk says no.
    fake_hash = "a" * 64
    tmp_db.execute(
        """INSERT INTO items(item_type, title, content_hash, metadata_source, metadata_confidence)
           VALUES ('paper', 'Ghost paper', ?, 'manual', 1.0)""",
        (fake_hash,),
    )
    r = client.get(f"/files/{fake_hash}")
    assert r.status_code == 404
