"""Tiered deduplication per plan §3.

The decision is a pure function of the candidate metadata + current DB state +
optional embedder and LLM judge. Mutation is done separately by ``act``, so
``dedup-scan`` (dry-run) and the ingest pipeline share the same decision code."""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from grimoire.embed.base import Embedder, l2_normalize, serialize_float32
from grimoire.embed.specter2 import format_item_text
from grimoire.models import Author, Metadata
from grimoire.search.semantic import search_items_by_embedding

log = logging.getLogger(__name__)


DedupOutcome = Literal["insert", "merge", "link", "skip"]
JudgeVerdict = Literal["same", "related", "different"]
JudgeFn = Callable[[Metadata, Metadata], JudgeVerdict | None]


# Threshold defaults from plan §3. Exposed here so tests / tools can tune.
TIER4_MERGE_SIM = 0.97
TIER4_JUDGE_LOW = 0.90
TIER4_JUDGE_HIGH = 0.97
TIER4_RELATED_NO_AUTHOR_SIM = 0.95
TIER4_NEIGHBORS = 10

# Plan §7 invariant 3: every directional relation has a symmetric inverse so
# queries from either side surface the link.
_INVERSE_RELATION: dict[str, str] = {
    "preprint_of": "published_as",
    "published_as": "preprint_of",
    "erratum_for": "corrected_by",
    "corrected_by": "erratum_for",
    "chapter_of": "contains_chapter",
    "contains_chapter": "chapter_of",
    "part_of": "contains_part",
    "contains_part": "part_of",
    "later_edition_of": "earlier_edition_of",
    "earlier_edition_of": "later_edition_of",
    "translates": "translated_from",
    "translated_from": "translates",
    "cites": "cited_by",
    "cited_by": "cites",
    "related": "related",  # symmetric
}

_ERRATUM_RE = re.compile(r"^\s*(erratum|correction|corrigendum)(\s+to)?\s*:?\s*", re.IGNORECASE)


@dataclass(slots=True)
class DedupDecision:
    outcome: DedupOutcome
    target_id: int | None
    reason: str
    relation: str | None = None
    confidence: float = 0.0


# ---------- decision ------------------------------------------------------


def decide(
    conn: sqlite3.Connection,
    candidate: Metadata,
    content_hash: str | None,
    *,
    item_embedder: Embedder | None = None,
    llm_judge: JudgeFn | None = None,
    exclude_item_id: int | None = None,
) -> DedupDecision:
    """Run the tiered algorithm. Does not mutate the DB.

    ``exclude_item_id`` is used by ``dedup-scan`` so an existing item isn't
    matched against itself when re-evaluating the corpus."""
    # Tier 1: deterministic identifier / hash match.
    t1 = _tier1(conn, candidate, content_hash, exclude_item_id)
    if t1:
        return t1

    # Tier 2: erratum / correction title pattern → LINK (never merge).
    t2 = _tier2_erratum(conn, candidate, exclude_item_id)
    if t2:
        return t2

    # Tier 3: arXiv entry with linked published DOI → preprint_of LINK.
    t3 = _tier3_arxiv_linked(conn, candidate, exclude_item_id)
    if t3:
        return t3

    # Tier 4: semantic. Opt-in — needs a loaded embedder.
    if item_embedder is not None:
        t4 = _tier4_semantic(conn, candidate, item_embedder, llm_judge, exclude_item_id)
        if t4:
            return t4

    return DedupDecision("insert", None, "no_match")


def _tier1(
    conn: sqlite3.Connection,
    candidate: Metadata,
    content_hash: str | None,
    exclude: int | None,
) -> DedupDecision | None:
    def _by(col: str, val: str) -> DedupDecision | None:
        row = conn.execute(
            f"SELECT id FROM items WHERE {col}=? AND id IS NOT ? LIMIT 1",
            (val, exclude),
        ).fetchone()
        if row:
            return DedupDecision("merge", int(row["id"]), f"{col}_match", None, 1.0)
        return None

    if content_hash:
        hit = _by("content_hash", content_hash)
        if hit:
            hit.reason = "hash_match"
            return hit
    if candidate.doi:
        hit = _by("doi", candidate.doi)
        if hit:
            return hit
    if candidate.arxiv_id:
        hit = _by("arxiv_id", candidate.arxiv_id)
        if hit:
            return hit
    if candidate.isbn:
        hit = _by("isbn", candidate.isbn)
        if hit:
            return hit
    return None


def _tier2_erratum(
    conn: sqlite3.Connection, candidate: Metadata, exclude: int | None
) -> DedupDecision | None:
    if not candidate.title:
        return None
    m = _ERRATUM_RE.match(candidate.title)
    if not m:
        return None
    # Title suffix after the "Erratum to:" prefix usually names the original paper.
    suffix = candidate.title[m.end() :].strip().strip("\"“”'.").strip()
    if len(suffix) < 15:
        return None
    row = conn.execute(
        "SELECT id FROM items WHERE title LIKE ? AND id IS NOT ? LIMIT 1",
        (f"%{suffix}%", exclude),
    ).fetchone()
    if row:
        return DedupDecision("link", int(row["id"]), "erratum_title_match", "erratum_for", 0.9)
    return None


def _tier3_arxiv_linked(
    conn: sqlite3.Connection, candidate: Metadata, exclude: int | None
) -> DedupDecision | None:
    if not candidate.arxiv_id or not candidate.raw:
        return None
    linked_doi = candidate.raw.get("linked_doi")
    if not linked_doi:
        return None
    row = conn.execute(
        "SELECT id FROM items WHERE doi=? AND id IS NOT ? LIMIT 1",
        (linked_doi, exclude),
    ).fetchone()
    if row:
        return DedupDecision("link", int(row["id"]), "arxiv_linked_doi", "preprint_of", 1.0)
    return None


def _tier4_semantic(
    conn: sqlite3.Connection,
    candidate: Metadata,
    embedder: Embedder,
    llm_judge: JudgeFn | None,
    exclude: int | None,
) -> DedupDecision | None:
    if not candidate.title:
        return None

    text = format_item_text(candidate.title, candidate.abstract)
    vec = l2_normalize(embedder.encode([text]))[0]
    neighbors = search_items_by_embedding(conn, vec, limit=TIER4_NEIGHBORS)
    cand_keys = {a.normalized_key for a in candidate.authors if a.family_name}

    for n in neighbors:
        if exclude is not None and n.item_id == exclude:
            continue
        if _is_asserted_non_duplicate(conn, exclude, n.item_id):
            continue

        sim = _cosine_from_score(n.score)
        neighbor_keys = _author_keys(conn, n.item_id)
        overlap = len(cand_keys & neighbor_keys)

        if sim >= TIER4_MERGE_SIM and overlap >= 1:
            return DedupDecision("merge", n.item_id, f"embed_sim_{sim:.3f}", None, sim)

        if TIER4_JUDGE_LOW <= sim < TIER4_JUDGE_HIGH and overlap >= 1:
            if llm_judge is None:
                # Conservative: without a judge we can't resolve the ambiguous
                # band. Don't auto-merge. dedup-scan with LLM enabled can be run
                # later to revisit.
                continue
            neighbor_meta = _load_item_metadata(conn, n.item_id)
            if neighbor_meta is None:
                continue
            verdict = llm_judge(candidate, neighbor_meta)
            if verdict == "same":
                return DedupDecision("merge", n.item_id, "llm_judge_same", None, sim)
            if verdict == "related":
                return DedupDecision("link", n.item_id, "llm_judge_related", "related", sim)
            # 'different' or None → continue scanning neighbors

        if sim >= TIER4_RELATED_NO_AUTHOR_SIM and overlap == 0:
            return DedupDecision("link", n.item_id, f"related_no_overlap_{sim:.3f}", "related", sim)

    return None


# ---------- application ---------------------------------------------------


def apply_merge(conn: sqlite3.Connection, target_id: int, candidate: Metadata) -> None:
    """Fill missing fields on the target with values from the candidate (plan §3
    'Union metadata; prefer most-authoritative source'). The existing item is
    already the authoritative one; we only backfill gaps."""
    row = conn.execute(
        """SELECT title, abstract, publication_year, doi, arxiv_id, isbn,
                  venue, volume, issue, pages, series, series_number, edition, language,
                  metadata_source, metadata_confidence, metadata_json
           FROM items WHERE id=?""",
        (target_id,),
    ).fetchone()
    if row is None:
        return

    import json

    updates: dict[str, object] = {}
    for field, cand_val in [
        ("title", candidate.title),
        ("abstract", candidate.abstract),
        ("publication_year", candidate.publication_year),
        ("doi", candidate.doi),
        ("arxiv_id", candidate.arxiv_id),
        ("isbn", candidate.isbn),
        ("venue", candidate.venue),
        ("volume", candidate.volume),
        ("issue", candidate.issue),
        ("pages", candidate.pages),
        ("series", candidate.series),
        ("series_number", candidate.series_number),
        ("edition", candidate.edition),
        ("language", candidate.language),
    ]:
        existing = row[field]
        if (existing is None or existing == "") and cand_val not in (None, ""):
            updates[field] = cand_val

    if candidate.raw:
        merged: dict[str, object] = dict(json.loads(row["metadata_json"] or "{}"))
        merged.update(candidate.raw)
        updates["metadata_json"] = json.dumps(merged)

    if updates:
        assignments = ", ".join(f"{k}=?" for k in updates)
        conn.execute(
            f"UPDATE items SET {assignments} WHERE id=?",
            (*updates.values(), target_id),
        )

    _union_authors(conn, target_id, candidate.authors)


def apply_link(
    conn: sqlite3.Connection,
    subject_id: int,
    object_id: int,
    relation: str,
    confidence: float,
) -> None:
    """Insert a directional relation and its symmetric inverse (plan §7 inv. 3)."""
    inverse = _INVERSE_RELATION.get(relation)
    if inverse is None:
        raise ValueError(f"Unknown relation: {relation!r}")
    conn.execute(
        """INSERT OR IGNORE INTO item_relations(subject_id, relation, object_id, confidence)
           VALUES (?, ?, ?, ?)""",
        (subject_id, relation, object_id, confidence),
    )
    if inverse == relation and subject_id == object_id:
        return  # self-inverse on the same node — noop
    conn.execute(
        """INSERT OR IGNORE INTO item_relations(subject_id, relation, object_id, confidence)
           VALUES (?, ?, ?, ?)""",
        (object_id, inverse, subject_id, confidence),
    )


# ---------- helpers -------------------------------------------------------


def _cosine_from_score(score: float) -> float:
    """Convert search.semantic's score back to cosine similarity.

    Score is ``-L2_distance`` for unit-normalized vectors, and for those
    ``L2² = 2(1 - cos)`` → ``cos = 1 - d²/2``."""
    distance = -score
    return 1.0 - (distance * distance) / 2.0


def _author_keys(conn: sqlite3.Connection, item_id: int) -> set[str]:
    return {
        row["normalized_key"]
        for row in conn.execute(
            """SELECT a.normalized_key FROM item_authors ia
               JOIN authors a ON a.id = ia.author_id
               WHERE ia.item_id = ?""",
            (item_id,),
        )
    }


def _is_asserted_non_duplicate(conn: sqlite3.Connection, a_id: int | None, b_id: int) -> bool:
    if a_id is None:
        return False  # candidate has no id yet (ingest path)
    lo, hi = (a_id, b_id) if a_id < b_id else (b_id, a_id)
    row = conn.execute(
        "SELECT 1 FROM non_duplicate_pairs WHERE a_id=? AND b_id=?", (lo, hi)
    ).fetchone()
    return row is not None


def _load_item_metadata(conn: sqlite3.Connection, item_id: int) -> Metadata | None:
    row = conn.execute(
        "SELECT title, abstract, doi, arxiv_id, isbn FROM items WHERE id=?", (item_id,)
    ).fetchone()
    if row is None:
        return None
    authors_rows = conn.execute(
        """SELECT a.family_name, a.given_name
           FROM item_authors ia JOIN authors a ON a.id = ia.author_id
           WHERE ia.item_id=? ORDER BY ia.position""",
        (item_id,),
    ).fetchall()
    return Metadata(
        title=row["title"],
        abstract=row["abstract"],
        doi=row["doi"],
        arxiv_id=row["arxiv_id"],
        isbn=row["isbn"],
        authors=[
            Author(family_name=r["family_name"], given_name=r["given_name"]) for r in authors_rows
        ],
        source="manual",
    )


def _union_authors(conn: sqlite3.Connection, item_id: int, authors: list[Author]) -> None:
    """Insert any author that isn't already linked to this item. Order preserved
    for newly-added authors; existing authors keep their position."""
    existing = {
        row["author_id"]
        for row in conn.execute("SELECT author_id FROM item_authors WHERE item_id=?", (item_id,))
    }
    next_pos = conn.execute(
        "SELECT COALESCE(MAX(position), -1) + 1 FROM item_authors WHERE item_id=?",
        (item_id,),
    ).fetchone()[0]

    for author in authors:
        if not author.family_name:
            continue
        author_id = _upsert_author(conn, author)
        if author_id in existing:
            continue
        conn.execute(
            """INSERT OR IGNORE INTO item_authors(item_id, author_id, position, role)
               VALUES (?, ?, ?, 'author')""",
            (item_id, author_id, next_pos),
        )
        next_pos += 1


def _upsert_author(conn: sqlite3.Connection, author: Author) -> int:
    if author.orcid:
        row = conn.execute("SELECT id FROM authors WHERE orcid=?", (author.orcid,)).fetchone()
        if row:
            return int(row["id"])
    row = conn.execute(
        "SELECT id FROM authors WHERE normalized_key=? AND orcid IS ?",
        (author.normalized_key, author.orcid),
    ).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        """INSERT INTO authors(family_name, given_name, orcid, normalized_key)
           VALUES (?,?,?,?)""",
        (author.family_name, author.given_name, author.orcid, author.normalized_key),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


# Expose a helper for vec0 serialization (used by dedup-scan to compute the
# embedding for existing items that might not be indexed yet).
def serialize_embedding(vec) -> bytes:  # type: ignore[no-untyped-def]
    return serialize_float32(vec)
