from __future__ import annotations

import hashlib
from pathlib import Path

from grimoire.storage.cas import CAS


def test_store_roundtrip(tmp_path: Path) -> None:
    cas = CAS(tmp_path)
    data = b"hello world"
    h, target = cas.store(data)
    assert h == hashlib.sha256(data).hexdigest()
    assert target.exists()
    assert target.read_bytes() == data


def test_store_path_layout(tmp_path: Path) -> None:
    cas = CAS(tmp_path)
    data = b"abc"
    h, target = cas.store(data)
    # ab/cd/... layout
    rel = target.relative_to(tmp_path)
    assert rel.parts[0] == h[:2]
    assert rel.parts[1] == h[2:4]
    assert rel.parts[2] == h


def test_store_dedup_same_bytes(tmp_path: Path) -> None:
    cas = CAS(tmp_path)
    data = b"same"
    h1, p1 = cas.store(data)
    h2, p2 = cas.store(data)
    assert h1 == h2
    assert p1 == p2
    # Only one file on disk under that hash.
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 1


def test_exists(tmp_path: Path) -> None:
    cas = CAS(tmp_path)
    assert not cas.exists("deadbeef" * 8)
    h, _ = cas.store(b"payload")
    assert cas.exists(h)


def test_hash_of_path(tmp_path: Path) -> None:
    src = tmp_path / "in.bin"
    src.write_bytes(b"x" * 1000)
    assert CAS.hash_file(src) == hashlib.sha256(b"x" * 1000).hexdigest()
