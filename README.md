# Grimoire

Self-hosted literature server for a ~15k-item research library — papers, books,
reports, theses. Designed to replace Zotero for one user on one host, with
three clients:

- **Claude Code** via **MCP** — semantic + keyword search, snippet retrieval with
  page citations, related-work discovery, BibTeX export.
- **iPad / KOReader** via **OPDS 1.2** — browsing and reading.
- **Browser** via a **minimal web UI** — browse, search, download, cite.

The three things this does that Zotero doesn't:

1. **Semantic search** over papers and book chapters with self-hosted embeddings
   (SPECTER2 for paper-level, BGE-M3 for chunk-level passages).
2. **Embedding-based deduplication** — tiered algorithm (deterministic →
   semantic → LLM judge for the ambiguous band), with `preprint_of` /
   `later_edition_of` / `chapter_of` / `part_of` relations instead of blind
   merges.
3. **First-class book support** — editions, chapters, series, editor roles.

Non-goals are binding (plan §2): multi-user sync, browser extension, Word plugin,
public sharing, native mobile apps, CSL styles in v1, annotation round-trip,
author disambiguation beyond normalized-key + ORCID.

---

## Status

| Phase | What                                   | Status |
|-------|----------------------------------------|--------|
| 0     | Infra scaffold                         | done   |
| 1     | Ingestion core                         | done   |
| 2     | Search (FTS + embeddings + hybrid)     | done   |
| 3     | Deduplication                          | done (100% precision/recall on 100-item oracle) |
| 4     | MCP server                             | done (9 tools at `/mcp`) |
| 5     | OPDS + minimal web UI                  | done   |
| 6     | Book-specific features (chapters, editions) | done |
| 7     | Zotero migration (local SQLite)        | done (100% field-match on 50-item oracle) |

v1.0 pending three manual acceptance tests ([#1](https://github.com/deOliveira-R/grimoire/issues/1),
[#2](https://github.com/deOliveira-R/grimoire/issues/2),
[#3](https://github.com/deOliveira-R/grimoire/issues/3)).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Host (TrueNAS in production, any x86/arm box in dev)        │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────────┐             │
│  │ grimoire (API)  │◄──►│ translation-server   │             │
│  │ FastAPI +       │    │ (Zotero's, optional) │             │
│  │ MCP at /mcp +   │    └──────────────────────┘             │
│  │ OPDS at /opds + │                                         │
│  │ web UI at  /    │    ┌────────────┐                       │
│  └────────┬────────┘    │ GROBID     │ (optional, for PDF    │
│           │             │ (Docker)   │  header extraction)   │
│           │             └────────────┘                       │
│           │                                                  │
│     ┌─────▼──────┐       ┌────────────┐                      │
│     │ library.db │       │ files/     │ Content-addressed    │
│     │ SQLite +   │       │ ab/cd/hash │ (SHA-256)            │
│     │ FTS5 +     │       └────────────┘                      │
│     │ sqlite-vec │                                           │
│     └────────────┘                                           │
└──────────────────────────────────────────────────────────────┘
       │                │                    │
       │ MCP            │ OPDS               │ HTTP
       ▼                ▼                    ▼
  Claude Code      iPad / KOReader      Browser
```

Single process, single writer (`grimoire serve`). SQLite holds everything —
metadata, FTS indexes, and vector embeddings live in the same file.
Files-on-disk are content-addressed; the DB only stores the hash.

---

## Quick start

```bash
# install
pip install -e ".[dev,ingest,ml]"
grimoire init-db

# Bootstrap from a local Zotero library (one-shot)
grimoire migrate zotero --dry-run           # preview: how many items would import
grimoire migrate zotero                     # full import from ~/Documents/.../zotero.sqlite
# OR ingest individual files:
grimoire ingest path/to/paper.pdf
grimoire ingest path/to/books/              # recursive

# Compute embeddings (required for semantic search)
grimoire index                              # SPECTER2 (~8 min for 15k papers CPU-only)
                                            # BGE-M3 chunks runs afterwards; slower

# Serve everything
grimoire serve                              # 0.0.0.0:8000

# Search from the CLI (quick sanity check)
grimoire search "boron dilution" --mode hybrid
```

The Docker Compose stack (`docker compose up`) brings up the API + the
`translation-server` sidecar + a GROBID instance for better PDF header
extraction.

**LAN-only:** `grimoire serve` binds `0.0.0.0` with no auth. Safe behind your
home network; **not safe on the public internet**. Exposing externally requires
a reverse proxy with auth in front.

---

## The identifier chain

Every retrieval surface — MCP, OPDS, web UI — hands back item IDs that drill
straight into the source file:

```
chunk_embeddings.chunk_id   →  chunks(id, item_id, page, chunk_index, text)
                                                    │
                                                    ▼
             items(id, doi, arxiv_id, isbn, content_hash, …)
                                                    │
                                                    ▼
                              /files/{content_hash}   (CAS blob, PDF/EPUB)
```

- **`search`** (MCP / CLI / web UI) returns ranked items with the best-matching
  chunk snippet and its page number.
- **`get_snippets`** returns passage-level hits, each carrying `item_id`,
  `chunk_id`, `page`, and `text`.
- From any `item_id`: `get_item` (full metadata), `get_full_text(item_id,
  page=N)` (reconstruct body by page), `get_citation` (BibTeX), or
  `/files/{content_hash}` to download the original.

Two embedding layers:

- **SPECTER2 (768d)** over `title [SEP] abstract` → paper-level discovery
  ("which *paper* is relevant"). One vector per item.
- **BGE-M3 (1024d)** over ~400-word sentence-aware chunks → passage retrieval
  ("which *passage* answers this"). One vector per chunk, each tagged with its
  page.

Hybrid mode fuses keyword (FTS5) + semantic item-level + semantic chunk-level
via Reciprocal Rank Fusion (k=60).

---

## Data model (abridged — see [migrations/001_init.sql](migrations/001_init.sql))

**`items`** — one row per bibliographic entity. `item_type` ∈ {paper, book,
chapter, report, thesis, preprint, standard, patent, other}. Carries title,
abstract, year, DOI, arXiv ID, ISBN, venue, volume, issue, pages, series,
edition, language, `content_hash` → CAS, and `metadata_json` for source-specific
blobs.

**`item_authors`** — many-to-many with `role` ∈ {author, editor, translator,
advisor}. Editors attached to books, authors to chapters.

**`item_relations`** — typed edges between items:

| Relation             | Symmetric inverse       | Meaning                                    |
|----------------------|-------------------------|--------------------------------------------|
| `preprint_of`        | `published_as`          | arXiv preprint ↔ journal publication       |
| `erratum_for`        | `corrected_by`          | erratum / corrigendum                      |
| `chapter_of`         | `contains_chapter`      | chapter ↔ parent book                      |
| `part_of`            | `contains_part`         | volume ↔ multi-volume set                  |
| `later_edition_of`   | `earlier_edition_of`    | edition chain                              |
| `translates`         | `translated_from`       | translation pair                           |
| `cites`              | `cited_by`              | citation edge                              |
| `related`            | `related` (self)        | tier-4 semantic-similarity finding         |

Every directional relation has a persisted symmetric inverse — queries from
either side find the link (plan §7 invariant 3).

**`chunks` / `items_fts` / `chunks_fts` / `item_embeddings` /
`chunk_embeddings`** — the retrieval tables. FTS triggers keep FTS5 mirrors in
sync with content; vector tables live in `sqlite-vec`.

Schema is append-only (migrations in [migrations/](migrations/)). Changes to
`items` / `item_relations` / embedding tables require explicit approval per
plan §10 rule 5.

---

## CLI surface

```
grimoire init-db                               Apply pending migrations.
grimoire ingest PATH [--no-recursive]          PDF / EPUB → dedup → items + CAS.
grimoire index [--force] [--limit N]           Compute / refresh embeddings.
grimoire search QUERY [--mode hybrid|kw|sem]   Hybrid search from the terminal.
grimoire dedup-scan [--semantic]               Dry-run the tiered dedup.
grimoire serve [--host --port --reload]        FastAPI + MCP + OPDS + web UI.
grimoire mcp --transport stdio                 Just the MCP server (stdio).
grimoire migrate zotero [--dry-run] [--limit]  One-shot Zotero SQLite import.
```

---

## MCP tools (for Claude Code)

Mount at `http://<host>:8000/mcp`:

```json
{"mcpServers": {"grimoire": {"url": "http://<host>:8000/mcp"}}}
```

Nine tools, all in [src/grimoire/mcp/tools.py](src/grimoire/mcp/tools.py):

- `search` — hybrid / keyword / semantic, filterable by `item_type`.
- `get_item` — full metadata for an id.
- `get_full_text` — reconstruct body text, optionally scoped to a page.
- `get_snippets` — best-matching chunks, optionally scoped to an item.
- `list_related` — traverse `chapter_of` / `preprint_of` / etc.; kinds:
  `all | preprint_chain | structural | semantic | citations`.
- `get_citation` — BibTeX (only style in v1).
- `list_tags`, `list_collections`, `find_by_tag` — browse by facet.

---

## OPDS 1.2 feeds

For KOReader / Marvin / other OPDS clients:

- `/opds` — root navigation catalog
- `/opds/recent` — recent additions (acquisition feed with download links)
- `/opds/collections`, `/opds/collections/{id}` — by collection
- `/opds/tags`, `/opds/tags/{name}` — by tag
- `/opds/authors`, `/opds/authors/{id}` — by author
- `/opds/types/{item_type}` — by type
- `/opds/venues`, `/opds/venues/{name}` — by journal
- `/opds/years`, `/opds/years/{year}` — by year
- `/opds/search?q=...` — OpenSearch (keyword-only; `/opds/opensearch.xml` is
  the descriptor)
- `/files/{content_hash}` — stream CAS blob with Range support and a
  guessed-MIME content-type.

---

## Self-correcting invariants

Ten properties checked by the test suite on every commit (plan §7,
[CLAUDE.md](CLAUDE.md) #Self-correcting invariants):

1. `count(items) + count(merge_history) == count(ingest_log)` — conservation.
2. No two live items share `content_hash`.
3. Every directional relation has a symmetric inverse row.
4. `item_embeddings` is 768d; `chunk_embeddings` is 1024d.
5. Re-ingesting the same file is a no-op.
6. Dedup oracle: precision ≥ 0.95 AND recall ≥ 0.95.
7. `item_type='paper' AND doi IS NOT NULL` ⇒ `venue IS NOT NULL`
   (silent-Crossref-failure guard).
8. `count(items) == count(items_fts)`; same for chunks.
9. `non_duplicate_pairs`: `a_id < b_id`.
10. No orphan chunks.

Invariants 7 and 10 currently lack explicit tests
([#4](https://github.com/deOliveira-R/grimoire/issues/4)).

---

## Development

```bash
pytest                                  # full suite: ~250 tests, ~3s
pytest --ignore=tests/test_heavy_embed  # skip the tests that download models
mypy src/grimoire                       # strict mode, zero errors target
ruff check .                            # lint
```

Layout:

```
src/grimoire/
  app.py              FastAPI + MCP mount
  cli.py              typer
  config.py           env-driven Settings (GRIMOIRE_*)
  db.py               sqlite + sqlite-vec + migrations
  models.py           Metadata / Author / IngestResult dataclasses
  identify.py         DOI / arXiv / ISBN regex extraction
  ingest.py           file → metadata → dedup → insert pipeline
  dedup.py            tiered algorithm (+ edition detection)
  dedup_llm.py        Claude API judge (optional)
  book_split.py       chapter splitting + re-materialization
  chunk.py            sentence-aware chunker
  index.py            post-ingest embedding pipeline
  storage/cas.py      content-addressed store
  extract/            pdf + epub + grobid + book_structure
  resolve/            crossref + arxiv + openlibrary + llm_fallback
  search/             keyword + semantic + hybrid (RRF)
  embed/              specter2 + bge_m3 + stub
  mcp/                tools + server + citation
  migrate/            zotero → grimoire
  web/                opds + files + ui (Jinja2)
```

Oracles (manual / heavy runs) live in [tools/](tools/):

- `phase1_zotero_oracle.py` — 100-item metadata-match oracle
- `phase2_search_oracle.py` — 20-query recall@10 (needs queries file,
  [#3](https://github.com/deOliveira-R/grimoire/issues/3))
- `phase3_dedup_oracle.py` — dedup precision/recall
- `phase6_book_oracle.py` — 12-chapter book split + search
- `phase7_migration_oracle.py` — Zotero migration field-match

---

## Design sources

- [CLAUDE.md](CLAUDE.md) — rules-of-engagement for Claude Code working on this
  repo.
- Full implementation plan lives outside the repo (personal working doc);
  `plan §N` references throughout the code point at it.
- [Open issues](https://github.com/deOliveira-R/grimoire/issues) — v2 backlog,
  UI roadmap, manual-oracle acceptance tasks.

## License

MIT — see [pyproject.toml](pyproject.toml).
