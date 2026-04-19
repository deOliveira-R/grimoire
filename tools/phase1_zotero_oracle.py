"""Phase 1 oracle (plan §6): pick 100 items from Zotero with PDF + metadata,
ingest them into a fresh grimoire DB, then diff metadata field-by-field.

Target per plan: >=95% field-level match on title/year/DOI/authors.

Run:
    .venv/bin/python tools/phase1_zotero_oracle.py
"""

from __future__ import annotations

import argparse
import random
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ZOTERO_SQLITE = Path.home() / "Documents/Biblioteca/#Zotero/zotero.sqlite"
ZOTERO_STORAGE = Path.home() / "Documents/Biblioteca/#Zotero/storage"

SCRATCH = Path("/tmp/grimoire-oracle")
SCRATCH_ZOTERO_DB = SCRATCH / "zotero.sqlite"
SCRATCH_PDFS = SCRATCH / "pdfs"
GRIMOIRE_DATA = SCRATCH / "grimoire_data"


# ---------- Zotero extraction ---------------------------------------------


FIELDS = ["title", "date", "DOI", "publicationTitle", "volume", "issue", "pages"]


def _field_id(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT fieldID FROM fields WHERE fieldName=?", (name,)).fetchone()
    return int(row[0])


def _value(conn: sqlite3.Connection, item_id: int, field_id: int) -> str | None:
    row = conn.execute(
        """SELECT v.value FROM itemData d
           JOIN itemDataValues v ON v.valueID = d.valueID
           WHERE d.itemID=? AND d.fieldID=?""",
        (item_id, field_id),
    ).fetchone()
    return row[0] if row else None


def _year(date: str | None) -> int | None:
    if not date:
        return None
    for tok in date.split():
        if tok.isdigit() and 1000 <= int(tok) <= 2100:
            return int(tok)
    # YYYY-MM-DD
    head = date[:4]
    if head.isdigit():
        y = int(head)
        if 1000 <= y <= 2100:
            return y
    return None


def _authors(conn: sqlite3.Connection, item_id: int) -> list[tuple[str, str | None]]:
    rows = conn.execute(
        """SELECT c.lastName, c.firstName
           FROM itemCreators ic
           JOIN creators c ON c.creatorID = ic.creatorID
           JOIN creatorTypes ct ON ct.creatorTypeID = ic.creatorTypeID
           WHERE ic.itemID = ? AND ct.creatorType = 'author'
           ORDER BY ic.orderIndex""",
        (item_id,),
    ).fetchall()
    return [(last or "", first or None) for last, first in rows]


def _pdf_attachments(conn: sqlite3.Connection, item_id: int) -> list[tuple[str, str]]:
    """Return list of (item_key, filename) for PDF attachments of a parent item."""
    rows = conn.execute(
        """SELECT it.key, a.path
           FROM itemAttachments a
           JOIN items it ON it.itemID = a.itemID
           WHERE a.parentItemID = ? AND a.contentType = 'application/pdf'""",
        (item_id,),
    ).fetchall()
    out = []
    for key, path in rows:
        if not path:
            continue
        # Zotero stores as "storage:filename.pdf" for managed attachments
        if path.startswith("storage:"):
            fname = path[len("storage:") :]
            out.append((key, fname))
    return out


def pick_sample(conn: sqlite3.Connection, n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Return n sample items with full ground-truth metadata and at least one on-disk PDF."""
    fid = {f: _field_id(conn, f) for f in FIELDS}

    # Candidate item_ids: journalArticle with DOI present AND title present.
    rows = conn.execute(
        f"""
        SELECT i.itemID
        FROM items i
        JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
        WHERE it.typeName = 'journalArticle'
          AND EXISTS(SELECT 1 FROM itemData d WHERE d.itemID=i.itemID AND d.fieldID={fid["title"]})
          AND EXISTS(SELECT 1 FROM itemData d WHERE d.itemID=i.itemID AND d.fieldID={fid["DOI"]})
          AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
        """
    ).fetchall()
    all_ids = [r[0] for r in rows]
    random.seed(seed)
    random.shuffle(all_ids)

    sample = []
    for iid in all_ids:
        if len(sample) >= n:
            break
        pdfs = _pdf_attachments(conn, iid)
        if not pdfs:
            continue
        key, fname = pdfs[0]
        src = ZOTERO_STORAGE / key / fname
        if not src.exists():
            continue
        sample.append(
            {
                "zotero_item_id": iid,
                "title": _value(conn, iid, fid["title"]),
                "year": _year(_value(conn, iid, fid["date"])),
                "doi": _value(conn, iid, fid["DOI"]),
                "venue": _value(conn, iid, fid["publicationTitle"]),
                "authors": _authors(conn, iid),
                "pdf_path": src,
            }
        )
    return sample


# ---------- grimoire side -------------------------------------------------


def prep_scratch(sample: list[dict[str, Any]]) -> None:
    if SCRATCH_PDFS.exists():
        shutil.rmtree(SCRATCH_PDFS)
    SCRATCH_PDFS.mkdir(parents=True)
    for i, item in enumerate(sample):
        dst = SCRATCH_PDFS / f"zot_{i:04d}_{item['zotero_item_id']}.pdf"
        shutil.copy(item["pdf_path"], dst)
        item["scratch_path"] = dst

    if GRIMOIRE_DATA.exists():
        shutil.rmtree(GRIMOIRE_DATA)
    GRIMOIRE_DATA.mkdir(parents=True)


def ingest_sample() -> None:
    import os

    os.environ["GRIMOIRE_DATA_ROOT"] = str(GRIMOIRE_DATA)
    # Reload settings so new env var takes effect.
    import importlib

    from grimoire import config as cfg

    importlib.reload(cfg)

    from grimoire.db import apply_migrations, connect
    from grimoire.ingest import ingest_path

    conn = connect()
    apply_migrations(conn)
    ingest_path(conn, SCRATCH_PDFS, recursive=False)


# ---------- comparison ----------------------------------------------------


def _norm_text(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(s.strip().lower().split())


def _norm_doi(s: str | None) -> str:
    if not s:
        return ""
    return s.strip().lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")


def _author_set(authors: list[Any]) -> set[str]:
    """Normalize to set of 'lastname' (lowercased). Ignore first names for
    the oracle since they come from different sources and may vary in
    initials vs. full."""
    out = set()
    for a in authors:
        if isinstance(a, tuple):
            last, _first = a
        else:
            last = a
        if last:
            out.add(last.strip().lower())
    return out


def compare(sample: list[dict[str, Any]]) -> dict[str, Any]:
    import sqlite3 as _sql

    conn = _sql.connect(GRIMOIRE_DATA / "db" / "library.db")
    conn.row_factory = _sql.Row

    results = {
        "total": len(sample),
        "ingested": 0,
        "title_match": 0,
        "year_match": 0,
        "doi_match": 0,
        "author_jaccard_sum": 0.0,
        "author_perfect_match": 0,
        "venue_match": 0,
        "failures": [],
    }

    for item in sample:
        # Look up the grimoire item_id via ingest_log (source_path matches the resolved scratch path).
        log_row = conn.execute(
            "SELECT item_id FROM ingest_log WHERE source_path=? AND item_id IS NOT NULL ORDER BY id DESC LIMIT 1",
            (str(item["scratch_path"].resolve()),),
        ).fetchone()
        if not log_row or log_row["item_id"] is None:
            results["failures"].append(
                {"zot_id": item["zotero_item_id"], "reason": "no grimoire item"}
            )
            continue

        grim_id = log_row["item_id"]
        results["ingested"] += 1
        row = conn.execute(
            "SELECT title, publication_year, doi, venue FROM items WHERE id=?", (grim_id,)
        ).fetchone()

        if _norm_text(row["title"]) == _norm_text(item["title"]):
            results["title_match"] += 1
        if row["publication_year"] == item["year"]:
            results["year_match"] += 1
        if _norm_doi(row["doi"]) == _norm_doi(item["doi"]):
            results["doi_match"] += 1
        if _norm_text(row["venue"]) == _norm_text(item["venue"]):
            results["venue_match"] += 1

        grim_authors = conn.execute(
            """SELECT a.family_name
               FROM item_authors ia JOIN authors a ON a.id = ia.author_id
               WHERE ia.item_id=? ORDER BY ia.position""",
            (grim_id,),
        ).fetchall()
        grim_set = _author_set([r["family_name"] for r in grim_authors])
        zot_set = _author_set(item["authors"])
        if zot_set:
            inter = grim_set & zot_set
            uni = grim_set | zot_set
            jacc = len(inter) / len(uni) if uni else 1.0
            results["author_jaccard_sum"] += jacc
            if grim_set == zot_set:
                results["author_perfect_match"] += 1

    return results


def _sha_of(path: Path) -> str:
    # Only used as a fallback marker — we don't need cryptographic use here.
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- main ----------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--sample-size", type=int, default=100)
    parser.add_argument("--skip-ingest", action="store_true", help="reuse existing grimoire DB")
    args = parser.parse_args()

    SCRATCH.mkdir(exist_ok=True)
    if not SCRATCH_ZOTERO_DB.exists():
        print(f"copying Zotero DB → {SCRATCH_ZOTERO_DB}", flush=True)
        shutil.copy(ZOTERO_SQLITE, SCRATCH_ZOTERO_DB)

    print("picking sample...", flush=True)
    zot = sqlite3.connect(SCRATCH_ZOTERO_DB)
    sample = pick_sample(zot, args.sample_size)
    print(f"  picked {len(sample)} items with DOI + PDF", flush=True)
    if len(sample) < args.sample_size:
        print(f"  (requested {args.sample_size}; pool was smaller)", flush=True)

    if not args.skip_ingest:
        prep_scratch(sample)
        print(f"ingesting {len(sample)} PDFs → {GRIMOIRE_DATA}...", flush=True)
        t0 = time.perf_counter()
        ingest_sample()
        print(f"  ingested in {time.perf_counter() - t0:.1f}s", flush=True)
    else:
        for i, it in enumerate(sample):
            it["scratch_path"] = SCRATCH_PDFS / f"zot_{i:04d}_{it['zotero_item_id']}.pdf"

    print("comparing...", flush=True)
    r = compare(sample)

    n = r["total"]
    print()
    print(f"=== Phase 1 oracle results ({n} items) ===")
    print(f"  ingested:              {r['ingested']}/{n}")
    print(f"  title match:           {r['title_match']}/{n}  ({100 * r['title_match'] / n:.1f}%)")
    print(f"  year match:            {r['year_match']}/{n}  ({100 * r['year_match'] / n:.1f}%)")
    print(f"  DOI match:             {r['doi_match']}/{n}  ({100 * r['doi_match'] / n:.1f}%)")
    print(f"  venue match:           {r['venue_match']}/{n}  ({100 * r['venue_match'] / n:.1f}%)")
    print(
        f"  author perfect match:  {r['author_perfect_match']}/{n}  "
        f"({100 * r['author_perfect_match'] / n:.1f}%)"
    )
    if n:
        print(f"  author Jaccard avg:    {r['author_jaccard_sum'] / n:.3f}")
    if r["failures"]:
        print(f"  failures:              {len(r['failures'])}")
        for f in r["failures"][:5]:
            print(f"    - {f}")

    # Overall field-level match across title/year/DOI/authors (the plan's target)
    per_field_total = 4 * n  # title, year, doi, authors
    per_field_hit = r["title_match"] + r["year_match"] + r["doi_match"] + r["author_perfect_match"]
    if per_field_total:
        print()
        print(
            f"Overall field-level match (title+year+doi+authors): "
            f"{per_field_hit}/{per_field_total} = {100 * per_field_hit / per_field_total:.1f}%"
        )
        print("Plan target: ≥95%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
