from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from grimoire import db as db_module


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    """Fresh database with all migrations applied, isolated to tmp_path."""
    db_path = tmp_path / "library.db"
    monkeypatch.setattr("grimoire.config.settings.data_root", tmp_path)
    conn = db_module.connect(db_path)
    db_module.apply_migrations(conn)
    return conn
