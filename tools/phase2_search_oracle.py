"""Phase 2 oracle (plan §6): compare keyword / semantic / hybrid on known-relevant queries.

Queries are mined from the corpus itself: for each target paper we build two
queries, one keyword-friendly (title tokens) and one concept-friendly (abstract
tokens that don't appear in the title). Semantic retrieval should dominate on
the conceptual variant; keyword on the exact-term variant; hybrid should
be best on the combined average.

Prereq: run tools/phase1_zotero_oracle.py first so the corpus is ingested at
/tmp/grimoire-oracle/grimoire_data. This tool then builds item-level SPECTER2
embeddings (skipping BGE-M3 chunks — item-level is all the oracle needs).
"""

from __future__ import annotations

import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter

# Prep environment before importing grimoire so it points at the oracle DB.
os.environ["GRIMOIRE_DATA_ROOT"] = "/tmp/grimoire-oracle/grimoire_data"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from grimoire.db import connect
from grimoire.embed.base import l2_normalize, serialize_float32
from grimoire.embed.specter2 import Specter2Embedder, format_item_text
from grimoire.search import search_items

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "we",
    "our",
    "these",
    "those",
    "some",
    "can",
    "not",
    "no",
    "also",
    "such",
    "may",
    "been",
    "but",
    "if",
    "than",
    "very",
    "which",
    "however",
    "where",
    "when",
    "while",
    "between",
    "about",
    "into",
    "here",
    "their",
    "his",
    "her",
    "them",
    "they",
    "us",
    "you",
    "your",
    "among",
    "during",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9\-]+", (text or "").lower())


def content_words(text: str, exclude: set[str] = frozenset()) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and t not in exclude and len(t) > 2]


def build_embeddings(conn: sqlite3.Connection, embedder: Specter2Embedder) -> int:
    """Embed every item's title+abstract with SPECTER2 and store into item_embeddings.
    Returns number indexed."""
    rows = conn.execute(
        """SELECT i.id, i.title, i.abstract
           FROM items i
           LEFT JOIN item_embeddings e ON e.item_id = i.id
           WHERE e.item_id IS NULL""",
    ).fetchall()
    if not rows:
        return 0
    texts = [format_item_text(r["title"] or "", r["abstract"]) for r in rows]
    # Small batches to keep CPU memory bounded.
    batch = 16
    n = 0
    for i in range(0, len(texts), batch):
        chunk_texts = texts[i : i + batch]
        vecs = l2_normalize(embedder.encode(chunk_texts))
        for r, v in zip(rows[i : i + batch], vecs, strict=True):
            conn.execute(
                "INSERT INTO item_embeddings(item_id, embedding) VALUES (?, ?)",
                (r["id"], serialize_float32(v)),
            )
            n += 1
    return n


def pick_queries(conn: sqlite3.Connection, n: int, seed: int = 1337) -> list[dict]:
    """For each of n target items build one exact-term and one conceptual query."""
    rows = conn.execute(
        """SELECT id, title, abstract
           FROM items
           WHERE title IS NOT NULL
             AND abstract IS NOT NULL
             AND length(abstract) > 400""",
    ).fetchall()
    rnd = random.Random(seed)
    rnd.shuffle(rows)

    out = []
    for r in rows:
        if len(out) >= n:
            break
        title_tokens = content_words(r["title"])
        if len(title_tokens) < 4:
            continue
        abstract_tokens = content_words(r["abstract"], exclude=set(title_tokens))
        if len(abstract_tokens) < 8:
            continue

        # Exact-term query: the full title (plenty of FTS signal)
        q_exact = r["title"]
        # Conceptual query: deterministic picks from abstract, no title tokens
        # (forces semantic path; keyword can't match via title)
        abstract_unique = list(dict.fromkeys(abstract_tokens))  # dedup, preserve order
        q_concept = " ".join(abstract_unique[:10])

        out.append(
            {
                "item_id": r["id"],
                "title": r["title"],
                "q_exact": q_exact,
                "q_concept": q_concept,
            }
        )
    return out


def run_oracle(
    conn: sqlite3.Connection,
    queries: list[dict],
    embedder: Specter2Embedder,
    limit: int = 10,
) -> dict:
    """Run each query in all three modes; count recall@limit per mode per flavor."""
    # "recall@10" with a single target per query is equivalent to hit rate.
    hits = {
        ("exact", "keyword"): 0,
        ("exact", "semantic"): 0,
        ("exact", "hybrid"): 0,
        ("concept", "keyword"): 0,
        ("concept", "semantic"): 0,
        ("concept", "hybrid"): 0,
    }
    ranks: dict = Counter()
    latencies: dict = {k: [] for k in hits}

    for q in queries:
        target = q["item_id"]
        for flavor in ("exact", "concept"):
            qstr = q["q_exact"] if flavor == "exact" else q["q_concept"]
            for mode in ("keyword", "semantic", "hybrid"):
                t0 = time.perf_counter()
                results = search_items(
                    conn,
                    qstr,
                    mode=mode,
                    limit=limit,
                    item_embedder=embedder,
                )
                latencies[(flavor, mode)].append((time.perf_counter() - t0) * 1000)
                ids = [h.item_id for h in results]
                if target in ids:
                    hits[(flavor, mode)] += 1
                    ranks[(flavor, mode, ids.index(target) + 1)] += 1

    return {"hits": hits, "ranks": ranks, "latencies": latencies, "total": len(queries)}


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-n", "--queries", type=int, default=20)
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()

    print("opening grimoire DB...", flush=True)
    conn = connect()

    print("loading SPECTER2...", flush=True)
    t0 = time.perf_counter()
    emb = Specter2Embedder()
    emb.encode(["warm"])
    print(f"  warm in {time.perf_counter() - t0:.1f}s", flush=True)

    print("building missing item embeddings...", flush=True)
    t0 = time.perf_counter()
    new = build_embeddings(conn, emb)
    n_items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    n_emb = conn.execute("SELECT COUNT(*) FROM item_embeddings").fetchone()[0]
    print(
        f"  {new} new / {n_emb} total embeddings over {n_items} items "
        f"in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    print(f"picking {args.queries} query pairs...", flush=True)
    queries = pick_queries(conn, args.queries)
    print(f"  got {len(queries)} pairs", flush=True)
    if not queries:
        print("no items with sufficient abstract length to build queries from.")
        return 1

    print("running searches...", flush=True)
    r = run_oracle(conn, queries, emb, limit=args.limit)

    n = r["total"]
    h = r["hits"]
    print()
    print(f"=== Phase 2 oracle: recall@{args.limit} over {n} queries ===")
    print(f"  corpus size: {n_items} items")
    print()
    print("  exact-term queries (full title as query):")
    print(
        f"    keyword:   {h[('exact', 'keyword')]}/{n}  ({100 * h[('exact', 'keyword')] / n:.0f}%)"
    )
    print(
        f"    semantic:  {h[('exact', 'semantic')]}/{n}  ({100 * h[('exact', 'semantic')] / n:.0f}%)"
    )
    print(f"    hybrid:    {h[('exact', 'hybrid')]}/{n}  ({100 * h[('exact', 'hybrid')] / n:.0f}%)")
    print()
    print("  conceptual queries (abstract words NOT in title):")
    print(
        f"    keyword:   {h[('concept', 'keyword')]}/{n}  ({100 * h[('concept', 'keyword')] / n:.0f}%)"
    )
    print(
        f"    semantic:  {h[('concept', 'semantic')]}/{n}  ({100 * h[('concept', 'semantic')] / n:.0f}%)"
    )
    print(
        f"    hybrid:    {h[('concept', 'hybrid')]}/{n}  ({100 * h[('concept', 'hybrid')] / n:.0f}%)"
    )
    print()
    print("  latency (median ms, query-embedding + KNN + hydrate):")
    import statistics

    for (flavor, mode), lats in r["latencies"].items():
        if lats:
            print(
                f"    {flavor:8s} {mode:8s} p50={statistics.median(lats):6.1f}ms  p95={sorted(lats)[int(0.95 * len(lats))]:6.1f}ms"
            )

    # Per-plan expectations:
    plan_pass = (
        h[("exact", "hybrid")] >= h[("exact", "keyword")]
        and h[("exact", "hybrid")] >= h[("exact", "semantic")]
        and h[("concept", "semantic")] >= h[("concept", "keyword")]
    )
    print()
    print("Plan expectations:")
    print(
        f"  hybrid ≥ keyword and semantic on exact:   {'OK' if (h[('exact', 'hybrid')] >= h[('exact', 'keyword')] and h[('exact', 'hybrid')] >= h[('exact', 'semantic')]) else 'FAIL'}"
    )
    print(
        f"  semantic ≥ keyword on conceptual:          {'OK' if h[('concept', 'semantic')] >= h[('concept', 'keyword')] else 'FAIL'}"
    )
    print(f"  overall:                                   {'PASS' if plan_pass else 'FAIL'}")

    return 0 if plan_pass else 1


if __name__ == "__main__":
    sys.exit(main())
