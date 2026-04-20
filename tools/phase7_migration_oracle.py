"""Phase 7 oracle (plan §6): sample 50 items from the migrated grimoire DB,
diff them against the ground truth in Zotero.

Assumes ``grimoire migrate zotero`` has already been run (or runs it
itself via ``--migrate``). Compares title/year/DOI/author fields; flags
per-field match rates.

Run after migration:
    .venv/bin/python tools/phase7_migration_oracle.py
Run migration + oracle in one go:
    .venv/bin/python tools/phase7_migration_oracle.py --migrate --limit 50

The "migration oracle" is a regression check for ``grimoire/migrate/zotero.py``
— any future change that silently loses fields or authors should surface
here before it touches Rodrigo's real corpus."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import sys
from pathlib import Path

ZOTERO_SQLITE = Path.home() / "Documents/Biblioteca/#Zotero/zotero.sqlite"
ZOTERO_STORAGE = Path.home() / "Documents/Biblioteca/#Zotero/storage"
SCRATCH = Path("/tmp/grimoire-phase7-oracle")


# ---------- text normalization (shared with phase 1 oracle) -----------------


def _norm_text(s: str | None) -> str:
    return " ".join((s or "").strip().lower().split())


def _norm_doi(s: str | None) -> str:
    return (
        (s or "")
        .strip()
        .lower()
        .removeprefix("https://doi.org/")
        .removeprefix("http://doi.org/")
    )


# ---------- Zotero ground truth ---------------------------------------------


def _field_ids(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        r["fieldName"]: int(r["fieldID"])
        for r in conn.execute("SELECT fieldName, fieldID FROM fields").fetchall()
    }


def _field_value(conn: sqlite3.Connection, item_id: int, field_id: int) -> str | None:
    row = conn.execute(
        """SELECT v.value FROM itemData d
           JOIN itemDataValues v ON v.valueID = d.valueID
           WHERE d.itemID = ? AND d.fieldID = ?""",
        (item_id, field_id),
    ).fetchone()
    return row["value"] if row else None


def _year_of(date: str | None) -> int | None:
    if not date:
        return None
    head = date.strip()[:4]
    if head.isdigit() and 1000 <= int(head) <= 2100:
        return int(head)
    return None


def _authors_of(conn: sqlite3.Connection, item_id: int) -> set[str]:
    rows = conn.execute(
        """SELECT c.lastName FROM itemCreators ic
           JOIN creators c ON c.creatorID = ic.creatorID
           JOIN creatorTypes ct ON ct.creatorTypeID = ic.creatorTypeID
           WHERE ic.itemID = ? AND ct.creatorType = 'author'""",
        (item_id,),
    ).fetchall()
    return {(r["lastName"] or "").strip().lower() for r in rows if r["lastName"]}


# ---------- oracle -----------------------------------------------------------


def pick_ground_truth(
    zotero_path: Path,
    grimoire_db: Path,
    sample_size: int,
    seed: int = 42,
) -> list[dict]:
    """Sample from Zotero items *actually migrated* — ``metadata_json``
    carries the original Zotero item_id, so we can intersect the two sides.

    Ensures the oracle works with any migration subset (``--limit``) rather
    than assuming the full library is present."""
    from urllib.parse import quote

    grim = sqlite3.connect(grimoire_db)
    grim.row_factory = sqlite3.Row
    migrated_ids = [
        int(r["zotero_item_id"])
        for r in grim.execute(
            "SELECT json_extract(metadata_json, '$.zotero_item_id') AS zotero_item_id "
            "FROM items WHERE metadata_json IS NOT NULL "
            "AND json_extract(metadata_json, '$.zotero_item_id') IS NOT NULL"
        ).fetchall()
    ]
    grim.close()

    random.seed(seed)
    random.shuffle(migrated_ids)
    picks = migrated_ids[:sample_size]

    conn = sqlite3.connect(f"file:{quote(str(zotero_path))}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    fid = _field_ids(conn)
    sample = []
    for item_id in picks:
        sample.append(
            {
                "zotero_item_id": item_id,
                "title": _field_value(conn, item_id, fid["title"]),
                "year": _year_of(_field_value(conn, item_id, fid["date"])),
                "doi": _field_value(conn, item_id, fid.get("DOI") or 0),
                "venue": _field_value(conn, item_id, fid.get("publicationTitle") or 0),
                "authors": _authors_of(conn, item_id),
            }
        )
    conn.close()
    return sample


def compare(grimoire_db: Path, sample: list[dict]) -> dict:
    conn = sqlite3.connect(grimoire_db)
    conn.row_factory = sqlite3.Row

    results = {
        "total": len(sample),
        "found": 0,
        "title_match": 0,
        "year_match": 0,
        "doi_match": 0,
        "venue_match": 0,
        "author_perfect_match": 0,
        "author_jaccard_sum": 0.0,
        "missing": [],
    }

    for item in sample:
        row = conn.execute(
            "SELECT id, title, publication_year, doi, venue FROM items "
            "WHERE json_extract(metadata_json, '$.zotero_item_id') = ?",
            (item["zotero_item_id"],),
        ).fetchone()
        if row is None:
            results["missing"].append(item["zotero_item_id"])
            continue
        results["found"] += 1
        if _norm_text(row["title"]) == _norm_text(item["title"]):
            results["title_match"] += 1
        if row["publication_year"] == item["year"]:
            results["year_match"] += 1
        if _norm_doi(row["doi"]) == _norm_doi(item["doi"]):
            results["doi_match"] += 1
        if _norm_text(row["venue"]) == _norm_text(item["venue"]):
            results["venue_match"] += 1

        grim_auth = {
            r["family_name"].strip().lower()
            for r in conn.execute(
                """SELECT a.family_name FROM item_authors ia
                   JOIN authors a ON a.id = ia.author_id WHERE ia.item_id = ?""",
                (row["id"],),
            ).fetchall()
            if r["family_name"]
        }
        zot_auth = item["authors"]
        if zot_auth:
            inter = grim_auth & zot_auth
            uni = grim_auth | zot_auth
            results["author_jaccard_sum"] += len(inter) / len(uni) if uni else 1.0
            if grim_auth == zot_auth:
                results["author_perfect_match"] += 1
    conn.close()
    return results


def run_migration(sample_size: int) -> Path:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    data_root = SCRATCH / "grimoire_data"
    os.environ["GRIMOIRE_DATA_ROOT"] = str(data_root)

    import importlib

    from grimoire import config as cfg

    importlib.reload(cfg)

    from grimoire.db import apply_migrations, connect
    from grimoire.migrate.zotero import migrate

    conn = connect()
    apply_migrations(conn)
    report = migrate(
        conn,
        library_path=ZOTERO_SQLITE,
        storage_dir=ZOTERO_STORAGE,
        limit=sample_size,
    )
    print(
        f"migrated: inserted={report.inserted}, merged={report.merged}, "
        f"skipped_no_title={report.skipped_no_metadata}, "
        f"pdfs={report.pdf_attachments_stored}"
    )
    conn.close()
    return data_root / "db" / "library.db"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--sample-size", type=int, default=50)
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Run the Zotero migration before comparing.",
    )
    parser.add_argument(
        "--grimoire-db",
        type=Path,
        default=None,
        help="Path to an already-migrated grimoire library.db to compare against.",
    )
    args = parser.parse_args()

    if args.migrate:
        grimoire_db = run_migration(args.sample_size)
    elif args.grimoire_db is not None:
        grimoire_db = args.grimoire_db
    else:
        print("Either --migrate or --grimoire-db is required.", file=sys.stderr)
        return 2

    # Sample from items actually migrated — avoids the trap where --limit
    # picked IDs A..N but the oracle happened to draw B..M.
    sample = pick_ground_truth(ZOTERO_SQLITE, grimoire_db, args.sample_size)
    if not sample:
        print("No migrated items found to compare.", file=sys.stderr)
        return 1

    r = compare(grimoire_db, sample)
    n = r["total"]
    print()
    print(f"=== Phase 7 migration oracle ({n} items sampled) ===")
    print(f"  found in grimoire:     {r['found']}/{n}")
    print(f"  title match:           {r['title_match']}/{n}  ({100 * r['title_match'] / n:.1f}%)")
    print(f"  year match:            {r['year_match']}/{n}  ({100 * r['year_match'] / n:.1f}%)")
    print(f"  DOI match:             {r['doi_match']}/{n}  ({100 * r['doi_match'] / n:.1f}%)")
    print(f"  venue match:           {r['venue_match']}/{n}  ({100 * r['venue_match'] / n:.1f}%)")
    if n:
        avg = r["author_jaccard_sum"] / n
        print(f"  author perfect match:  {r['author_perfect_match']}/{n}")
        print(f"  author jaccard avg:    {avg:.3f}")
    if r["missing"]:
        print(f"  not migrated:          {r['missing'][:5]}{'...' if len(r['missing']) > 5 else ''}")

    # Plan target: ≥95% per-field. Migration is lossless-ish (reads Zotero
    # directly, skips the Crossref resolve roundtrip) so this should be
    # ≥99% on well-formed items.
    ok = r["title_match"] >= 0.95 * n and r["doi_match"] >= 0.95 * n and r["year_match"] >= 0.95 * n
    print()
    print(f"Plan target (≥95% on title/year/DOI): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
