"""Stream CAS blobs for OPDS download links.

``GET /files/{content_hash}`` looks up the item by hash, maps it to the CAS
on-disk path, and returns it with an inferred MIME type. Range requests
work out of the box via Starlette's ``FileResponse``."""

from __future__ import annotations

import mimetypes
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from grimoire.config import settings
from grimoire.db import apply_migrations, connect
from grimoire.storage.cas import CAS

router = APIRouter(tags=["files"])

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _db() -> Iterator[sqlite3.Connection]:
    conn = connect()
    apply_migrations(conn)
    try:
        yield conn
    finally:
        conn.close()


def _guess_mime(file_path: str | None) -> str:
    if file_path:
        guessed, _ = mimetypes.guess_type(file_path)
        if guessed:
            return guessed
    return "application/octet-stream"


@router.get("/files/{content_hash}")
def download(
    content_hash: str,
    conn: sqlite3.Connection = Depends(_db),
) -> FileResponse:
    if not _HASH_RE.match(content_hash):
        raise HTTPException(status_code=400, detail="malformed content hash")

    row = conn.execute(
        "SELECT title, file_path FROM items WHERE content_hash = ? LIMIT 1",
        (content_hash,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="no item with that content hash")

    cas = CAS(settings.files_root)
    path = cas.path_for_hash(content_hash)

    # Defense-in-depth: reject anything that would escape the CAS root. Hash
    # regex already prevents traversal, but this catches symlink shenanigans
    # or a future CAS layout change that forgets this invariant.
    try:
        resolved = path.resolve(strict=True)
        root = Path(settings.files_root).resolve()
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="file missing from CAS")

    mime = _guess_mime(row["file_path"])
    # Filename for the download dialog: prefer the original basename, fall
    # back to the hash + guessed extension so we don't dump bare hashes on
    # the user's reader.
    ext = mimetypes.guess_extension(mime) or ""
    if row["file_path"]:
        download_name = Path(row["file_path"]).name
    else:
        download_name = f"{content_hash[:12]}{ext}"

    return FileResponse(path=resolved, media_type=mime, filename=download_name)
