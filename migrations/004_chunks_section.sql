-- Section-aware chunking (plan §6 Phase 2 refinement — post-v1 issue #17).
--
-- ``chunks.section`` carries the classified section type derived from the
-- GROBID TEI heading at index time: one of
--   'introduction', 'methods', 'results', 'discussion', 'conclusion', 'other'
-- or NULL for chunks that predate this migration (per-page fallback path
-- for items without a TEI artifact).
--
-- NULL is treated as "unknown section, don't filter out" by MCP search.

ALTER TABLE chunks ADD COLUMN section TEXT;

CREATE INDEX idx_chunks_section ON chunks(section) WHERE section IS NOT NULL;
