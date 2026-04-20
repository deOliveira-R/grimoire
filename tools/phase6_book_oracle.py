"""Phase 6 oracle (plan §6): book → chapter-split → index → search.

Builds a synthetic 12-chapter PDF (2 editors, chapter-specific content),
runs it through ``grimoire ingest``, then asserts:

  * 1 book item + 12 chapter items are present
  * Each chapter has a ``chapter_of`` relation back to the book (and the
    book has the symmetric ``contains_chapter`` inverses)
  * Running ``grimoire index`` embeds every chapter
  * Searching for a chapter-specific phrase surfaces *that* chapter first,
    not the parent book

Synthetic fixture so this can run offline (Rodrigo's real edited volume is
Phase 7 territory once migration is wired up).

Run:
    .venv/bin/python tools/phase6_book_oracle.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

SCRATCH = Path("/tmp/grimoire-phase6-oracle")
CHAPTERS = [
    ("Introduction to reactor physics", "neutron economy overview"),
    ("Cross sections and reaction rates", "microscopic cross section tabulation"),
    ("Diffusion theory basics", "one-group diffusion equation derivation"),
    ("Neutron slowing down", "lethargy Fermi age continuous slowing down model"),
    ("Resonance absorption", "resolved resonance region unresolved Doppler broadening"),
    ("Thermalization", "Maxwell Boltzmann spectrum scattering kernels"),
    ("Criticality and k-effective", "multiplication factor delayed neutron precursors"),
    ("Reactor kinetics", "prompt jump approximation point kinetics equations"),
    ("Reactivity feedback", "moderator temperature coefficient void coefficient"),
    ("Fuel depletion and burnup", "fission product poisons xenon samarium transients"),
    ("Boron dilution transients in PWR", "homogeneous heterogeneous mixing scenarios"),
    ("Decay heat and residual power", "ANSI 5.1 standard cooling curves"),
]


# ---------- synthetic book construction ---------------------------------------


def build_book_pdf(path: Path) -> None:
    """Produce a 24-page PDF with 2 pages per chapter and a TOC."""
    import pymupdf

    doc = pymupdf.open()
    chapter_page_starts: list[int] = []
    for i, (title, body) in enumerate(CHAPTERS):
        # 2 pages per chapter
        chapter_page_starts.append(len(doc) + 1)
        for page_offset in range(2):
            page = doc.new_page()
            if page_offset == 0:
                page.insert_text((72, 72), f"Chapter {i + 1}: {title}", fontsize=14)
                page.insert_text((72, 96), body, fontsize=11)
            else:
                page.insert_text(
                    (72, 72),
                    f"More text about {title.lower()} — page 2",
                    fontsize=11,
                )
    doc.set_toc([[1, title, start] for (title, _), start in zip(CHAPTERS, chapter_page_starts)])
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()


# ---------- oracle checks -----------------------------------------------------


def run() -> int:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    pdf_path = SCRATCH / "reactor_handbook.pdf"
    book_title = "Reactor Handbook (2nd ed.)"

    build_book_pdf(pdf_path)
    print(f"built synthetic book → {pdf_path} ({len(CHAPTERS)} chapters × 2 pages)")

    os.environ["GRIMOIRE_DATA_ROOT"] = str(SCRATCH / "grimoire_data")
    import importlib

    from grimoire import config as cfg

    importlib.reload(cfg)

    from grimoire.db import apply_migrations, connect
    from grimoire.ingest import ingest_file
    from grimoire.models import Author, Metadata

    conn = connect()
    apply_migrations(conn)

    # Force the ingest resolver to return a book with 2 editors (skips Crossref)
    from grimoire import ingest as ingest_module

    def _fake_resolve(_: Path) -> Metadata:
        return Metadata(
            title=book_title,
            abstract="A synthetic nuclear reactor physics handbook.",
            publication_year=2024,
            item_type="book",
            edition="2",
            authors=[
                Author(family_name="Editor1", given_name="Alice"),
                Author(family_name="Editor2", given_name="Bob"),
            ],
            source="manual",
            confidence=1.0,
        )

    ingest_module._resolve_metadata = _fake_resolve  # type: ignore[attr-defined]

    t0 = time.perf_counter()
    result = ingest_file(conn, pdf_path)
    t_ingest = time.perf_counter() - t0
    print(f"ingest: {result.outcome} in {t_ingest:.2f}s (book item_id={result.item_id})")

    # ---- structural checks ----
    book_id = result.item_id
    assert book_id is not None
    n_chapters = conn.execute(
        "SELECT COUNT(*) AS n FROM items WHERE item_type='chapter'"
    ).fetchone()["n"]
    print(f"chapter items created: {n_chapters} (expected {len(CHAPTERS)})")
    structure_ok = n_chapters == len(CHAPTERS)

    rel_forward = conn.execute(
        "SELECT COUNT(*) AS n FROM item_relations WHERE relation='chapter_of' AND object_id=?",
        (book_id,),
    ).fetchone()["n"]
    rel_inverse = conn.execute(
        "SELECT COUNT(*) AS n FROM item_relations WHERE relation='contains_chapter' AND subject_id=?",
        (book_id,),
    ).fetchone()["n"]
    print(f"chapter_of relations: {rel_forward}; contains_chapter inverses: {rel_inverse}")
    relations_ok = rel_forward == len(CHAPTERS) and rel_inverse == len(CHAPTERS)

    # ---- index every chapter and verify embeddings exist ----
    print("indexing (hashed stub embedders)...")
    import numpy as np

    from grimoire.index import index_all

    class HashedStub:
        """Deterministic per-text embedder — each text hashes to a unique vec.

        Avoids downloading the real SPECTER2/BGE-M3 models; good enough for
        structural checks (embedding dim + one vec per chapter)."""

        def __init__(self, dim: int) -> None:
            self.dim = dim

        def encode(self, texts: list[str]) -> np.ndarray:
            out = np.zeros((len(texts), self.dim), dtype=np.float32)
            for i, text in enumerate(texts):
                r = np.random.default_rng(abs(hash(text)) % (2**31))
                out[i] = r.standard_normal(self.dim).astype(np.float32)
            return out

    item_emb = HashedStub(dim=768)
    chunk_emb = HashedStub(dim=1024)
    results = index_all(conn, item_embedder=item_emb, chunk_embedder=chunk_emb)
    indexed_chapters = [
        r
        for r in results
        if conn.execute("SELECT item_type FROM items WHERE id=?", (r.item_id,)).fetchone()[
            "item_type"
        ]
        == "chapter"
    ]
    print(
        f"chapters indexed: {len(indexed_chapters)}  "
        f"(total chunks: {sum(r.chunks for r in indexed_chapters)})"
    )
    per_chapter_embeddings = conn.execute(
        """SELECT COUNT(*) AS n FROM items i
           JOIN item_embeddings e ON e.item_id = i.id
           WHERE i.item_type='chapter'"""
    ).fetchone()["n"]
    embeddings_ok = per_chapter_embeddings == len(CHAPTERS)
    print(
        f"chapter embeddings: {per_chapter_embeddings}/{len(CHAPTERS)} "
        f"{'OK' if embeddings_ok else 'MISSING'}"
    )

    # ---- chunk search: chapter-body phrase surfaces that chapter's chunk ----
    from grimoire.search.keyword import search_chunks

    # "homogeneous heterogeneous" only appears in chapter 11's body text.
    snippets = search_chunks(conn, "homogeneous heterogeneous mixing scenarios", limit=5)
    ch11_title = CHAPTERS[10][0]
    ch11_row = conn.execute(
        "SELECT id FROM items WHERE item_type='chapter' AND title=?", (ch11_title,)
    ).fetchone()
    ch11_id = int(ch11_row["id"]) if ch11_row else -1
    top_item_id = snippets[0].item_id if snippets else None
    search_ok = bool(snippets) and top_item_id == ch11_id
    print(
        f"chapter-specific chunk search: ch11 id={ch11_id}, top snippet's item_id={top_item_id} "
        f"{'OK' if search_ok else 'WRONG'}"
    )

    # ---- editors were kept on the book, not on chapters ----
    book_authors = conn.execute(
        """SELECT a.family_name FROM item_authors ia
           JOIN authors a ON a.id=ia.author_id
           WHERE ia.item_id=?""",
        (book_id,),
    ).fetchall()
    chapter_author_count = conn.execute(
        """SELECT COUNT(*) AS n FROM item_authors ia
           JOIN items i ON i.id=ia.item_id
           WHERE i.item_type='chapter'"""
    ).fetchone()["n"]
    authors_ok = len(book_authors) == 2 and chapter_author_count == 0
    print(
        f"book authors: {[r['family_name'] for r in book_authors]}  "
        f"chapter-level author rows: {chapter_author_count}  "
        f"{'OK' if authors_ok else 'WRONG'}"
    )

    print()
    print("=== Phase 6 oracle summary ===")
    print(f"  structural (13 items + relations): {'PASS' if structure_ok and relations_ok else 'FAIL'}")
    print(f"  per-chapter embeddings:            {'PASS' if embeddings_ok else 'FAIL'}")
    print(f"  chapter-specific search:           {'PASS' if search_ok else 'FAIL'}")
    print(f"  author/editor split:               {'PASS' if authors_ok else 'FAIL'}")

    all_pass = structure_ok and relations_ok and embeddings_ok and search_ok and authors_ok
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run())
