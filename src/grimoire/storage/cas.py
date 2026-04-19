"""Content-addressed file store. Layout: ``{root}/{ab}/{cd}/{fullhash}``.

Two writers of the same bytes converge on the same path, which is what makes
re-ingest of the same file a no-op on disk."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


class CAS:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(_CHUNK):
                h.update(chunk)
        return h.hexdigest()

    def path_for_hash(self, h: str) -> Path:
        return self.root / h[:2] / h[2:4] / h

    def exists(self, h: str) -> bool:
        return self.path_for_hash(h).exists()

    def store(self, data: bytes) -> tuple[str, Path]:
        h = self.hash_bytes(data)
        target = self.path_for_hash(h)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return h, target

    def store_file(self, src: Path) -> tuple[str, Path]:
        h = self.hash_file(src)
        target = self.path_for_hash(h)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(src.read_bytes())
        return h, target
