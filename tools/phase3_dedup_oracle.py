"""Phase 3 oracle (plan §6): precision + recall for the dedup algorithm.

Methodology
-----------
Zotero deduplicates by DOI itself, so we can't just mine known-duplicate pairs
from it. Instead we synthesize duplicates by byte-perturbing an already-ingested
PDF — the hash changes but the embedded DOI doesn't, forcing tier-1 to decide
via ``doi_match`` rather than ``hash_match``. That's the real test of tier-1
semantics.

Precision is measured the other way: re-evaluating every ingested item against
the rest of the corpus (with ``exclude_item_id`` set). Any ``merge`` or
``link`` decision is a spurious match — unless it's a legitimate
erratum/preprint pair, which we can't automatically distinguish from a false
positive. Report all of them and let the user eyeball the list.

Plan target: precision ≥ 0.95 AND recall ≥ 0.95.

Run:
    GRIMOIRE_CROSSREF_MAILTO=<email> python tools/phase3_dedup_oracle.py -n 200 --dups 50
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Must be set before `grimoire.config` imports anything.
os.environ["GRIMOIRE_DATA_ROOT"] = "/tmp/grimoire-oracle/grimoire_data"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sqlite3

from phase1_zotero_oracle import (
    SCRATCH,
    SCRATCH_ZOTERO_DB,
    ZOTERO_SQLITE,
    ingest_sample,
    pick_sample,
    prep_scratch,
)

from grimoire import dedup, ingest
from grimoire.db import apply_migrations, connect
from grimoire.models import Author, Metadata

DUP_DIR = SCRATCH / "perturbed_pdfs"


def _perturb(src: Path, dst: Path) -> None:
    """Append a null byte to the end of the PDF. The content_hash changes,
    the DOI in the header text does not. PyMuPDF tolerates trailing garbage."""
    dst.write_bytes(src.read_bytes() + b"\x00")


def build_corpus(sample_size: int) -> None:
    SCRATCH.mkdir(exist_ok=True)
    if not SCRATCH_ZOTERO_DB.exists():
        print(f"copying Zotero DB → {SCRATCH_ZOTERO_DB}", flush=True)
        shutil.copy(ZOTERO_SQLITE, SCRATCH_ZOTERO_DB)
    zot = sqlite3.connect(SCRATCH_ZOTERO_DB)
    sample = pick_sample(zot, sample_size)
    print(f"  picked {len(sample)} items", flush=True)
    prep_scratch(sample)
    print(f"ingesting {len(sample)} PDFs...", flush=True)
    ingest_sample()


def recall_test(n_dup: int) -> tuple[int, int]:
    """For n_dup ingested items, make a perturbed copy (different hash, same DOI)
    and ingest it. Every single one should return outcome='merged' via tier-1
    DOI. Returns (merged_count, total)."""
    if DUP_DIR.exists():
        shutil.rmtree(DUP_DIR)
    DUP_DIR.mkdir()

    conn = connect()
    apply_migrations(conn)
    rows = conn.execute(
        "SELECT id, file_path FROM items WHERE content_hash IS NOT NULL LIMIT ?",
        (n_dup,),
    ).fetchall()

    # Map item_id → original CAS path on disk.
    from grimoire.config import settings

    merged_count = 0
    for i, row in enumerate(rows):
        original_cas = Path(settings.files_root) / row["file_path"]
        if not original_cas.exists():
            print(f"  skip item {row['id']}: CAS file missing", flush=True)
            continue
        perturbed = DUP_DIR / f"dup_{i:04d}_{row['id']}.pdf"
        _perturb(original_cas, perturbed)

        result = ingest.ingest_file(conn, perturbed)
        if result.outcome == "merged" and result.item_id == row["id"]:
            merged_count += 1
        else:
            print(
                f"  MISS item_id={row['id']}: got outcome={result.outcome!r} "
                f"target={result.item_id!r} reason={result.reason!r}",
                flush=True,
            )

    return merged_count, len(rows)


def precision_test(limit: int | None = None) -> dict[str, int]:
    """Re-evaluate every item against the rest of the corpus with exclude_item_id=self.
    Any merge/link is a potential false positive (may be legitimate preprint/erratum).
    Returns counts by outcome + the list of flagged pairs."""
    conn = connect()

    # Tier-4 would need a loaded embedder. Opt-in via env var for this oracle.
    emb = None
    if os.environ.get("GRIMOIRE_ORACLE_SEMANTIC", "0") == "1":
        from grimoire.embed.specter2 import Specter2Embedder

        emb = Specter2Embedder()
        emb.encode(["warm"])

    q = "SELECT id, title, abstract, doi, arxiv_id, isbn, content_hash FROM items ORDER BY id"
    if limit:
        q += f" LIMIT {int(limit)}"
    items = conn.execute(q).fetchall()

    counts = {"insert": 0, "merge": 0, "link": 0, "skip": 0}
    flagged: list[tuple[int, str, int, str, float]] = []

    for row in items:
        authors = [
            Author(family_name=r["family_name"], given_name=r["given_name"])
            for r in conn.execute(
                """SELECT a.family_name, a.given_name FROM item_authors ia
                   JOIN authors a ON a.id = ia.author_id
                   WHERE ia.item_id=? ORDER BY ia.position""",
                (row["id"],),
            )
        ]
        cand = Metadata(
            title=row["title"],
            abstract=row["abstract"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            isbn=row["isbn"],
            authors=authors,
        )
        decision = dedup.decide(
            conn,
            cand,
            content_hash=row["content_hash"],
            item_embedder=emb,
            exclude_item_id=int(row["id"]),
        )
        counts[decision.outcome] += 1
        if decision.outcome in {"merge", "link"} and decision.target_id is not None:
            flagged.append(
                (
                    int(row["id"]),
                    decision.outcome,
                    int(decision.target_id),
                    decision.reason,
                    decision.confidence,
                )
            )

    print(f"\nflagged {len(flagged)} potential pairs:")
    for src_id, outcome, tgt_id, reason, conf in flagged[:20]:
        src = conn.execute("SELECT title FROM items WHERE id=?", (src_id,)).fetchone()["title"]
        tgt = conn.execute("SELECT title FROM items WHERE id=?", (tgt_id,)).fetchone()["title"]
        print(f"  [{outcome.upper()}] {src_id}→{tgt_id}  reason={reason}  conf={conf:.2f}")
        print(f"    src: {src[:90]!r}")
        print(f"    tgt: {tgt[:90]!r}")
    if len(flagged) > 20:
        print(f"  ... and {len(flagged) - 20} more")

    return {**counts, "flagged": len(flagged)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--corpus-size", type=int, default=200)
    p.add_argument("--dups", type=int, default=50, help="Number of perturbed-copy merge tests")
    p.add_argument("--skip-ingest", action="store_true")
    args = p.parse_args()

    if not args.skip_ingest:
        build_corpus(args.corpus_size)

    merged, total = 0, 0
    if args.dups > 0:
        print("\n== Recall: perturbed-copy MERGE via tier-1 DOI ==")
        merged, total = recall_test(args.dups)
        pct = (100 * merged / total) if total else 0.0
        print(f"  {merged}/{total} merged  ({pct:.1f}%)")
    else:
        print("\n== Recall test skipped (--dups 0) ==")

    print("\n== Precision: re-scan every item, report any merge/link ==")
    stats = precision_test(limit=args.corpus_size)
    total_items = sum(stats[k] for k in ("insert", "merge", "link", "skip"))
    print()
    print(f"  insert: {stats['insert']}")
    print(f"  merge:  {stats['merge']}    <- plan §3: these are the real false-positive risk")
    print(f"  link:   {stats['link']}    <- related-paper surfacing, expected, not counted as FP")
    print(f"  skip:   {stats['skip']}")
    # Plan §3: "False positives (incorrect merges) matter more than false negatives."
    # Precision here counts only incorrect merges; legitimate `related` links don't.
    fp_rate = stats["merge"] / total_items if total_items else 0.0
    print(f"  merge-FP rate: {100 * fp_rate:.1f}%")

    recall = (merged / total) if total else None
    precision = 1.0 - fp_rate
    print()
    if recall is not None:
        print(f"Recall    ≈ {recall:.3f}  (plan target: ≥ 0.95)")
    print(f"Precision ≈ {precision:.3f}  (plan target: ≥ 0.95)")
    print(
        "Note: flagged precision is an UPPER BOUND — some flags may be legitimate "
        "preprint/erratum relations. Inspect the list before concluding."
    )

    ok = (recall is None or recall >= 0.95) and precision >= 0.95
    print()
    print(f"Overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
