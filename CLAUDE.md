# Grimoire — working notes for Claude Code

Self-hosted literature server. Full design and rationale live in the implementation plan (the working document Rodrigo keeps alongside this repo).

## Rules of engagement

1. **Read the whole plan before writing code.** Oracle checks per phase are the contract.
2. **Finish each phase completely before starting the next.** Infrastructure debt compounds.
3. **Write oracle checks first each phase.** They are the acceptance criteria.
4. **Bias toward deletion.** Speculative modules — don't write them.
5. **Ask before changing the data model.** Schema changes to `items`, `item_relations`, or the embedding tables need explicit approval in a separate turn.
6. **Use the existing Zotero library as ground truth** (available at `../zotero/`).
7. **Do not build unrequested features.** Non-goals section of the plan is binding.
8. **Performance targets are hardware-relative.** X9SCM-F, CPU-only, SATA dataset.

## Phase map

| Phase | What | Status |
|-------|------|--------|
| 0 | Infra scaffold | done |
| 1 | Ingestion core | done (deterministic dedup only; real MERGE in Phase 3) |
| 2 | Search (FTS + embeddings + hybrid) | done (SPECTER2 + BGE-M3 lazy; RRF k=60) |
| 3 | Deduplication (heart of the project) | done (tiered + GROBID pre-extractor; 100% precision/recall on 100-item oracle) |
| 4 | MCP server | done (9 tools mounted at /mcp via FastMCP streamable-http) |
| 5 | OPDS + minimal web UI | done (Atom 1.2 feeds + /files CAS streamer + Jinja2 browse/detail UI; upload/submit-url deferred to v1.1) |
| 6 | Book-specific features | done (PDF/EPUB chapter split via TOC/spine; tier-4 edition detection; Crossref edition field) |
| 7 | Zotero migration | done (`grimoire migrate zotero` reads local SQLite; 100% field-match on 50-item oracle) |

## Run

```bash
pip install -e ".[dev,ingest]"
grimoire init-db
grimoire serve --reload            # local dev
grimoire migrate zotero --dry-run  # preview the Zotero import
grimoire migrate zotero            # full bootstrap from ~/.../zotero.sqlite
pytest                             # oracle checks
docker compose up                  # full stack (api + translation-server)
```

## Deployment

**LAN-only** for v1. `grimoire serve` binds 0.0.0.0 with no auth — safe behind the home network, not safe on the public internet. Exposing /opds, /files, or the web UI externally requires a reverse proxy with basic auth or OAuth in front.

## Self-correcting invariants (plan §7)

These must hold at all times and are enforced in tests:

1. `count(items) + count(merge_history) == count(ingest_log)`
2. No two live items share `content_hash`
3. Symmetric relations: `preprint_of(a,b) ↔ published_as(b,a)` etc.
4. Embedding dims: `item_embeddings = 768`, `chunk_embeddings = 1024`
5. Re-ingesting the same file is a no-op
6. Dedup oracle set: precision ≥ 0.95 AND recall ≥ 0.95
7. `item_type='paper' AND doi IS NOT NULL` ⇒ `venue IS NOT NULL`
8. `count(items) == count(items_fts)`; `count(chunks) == count(chunks_fts)`
9. `non_duplicate_pairs`: `a_id < b_id`
10. No orphan chunks
