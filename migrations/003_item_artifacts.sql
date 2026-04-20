-- Per-item derived artifacts (plan §6 — post-v1 strategic add).
--
-- Each item can now carry several files:
--   * 'primary'      the original PDF / EPUB (what items.content_hash already points at)
--   * 'grobid_tei'   GROBID-extracted TEI XML (structured sections + references)
--   * 'ocr_text'     plain-text fallback for scanned PDFs (deferred, #13)
--   * 'extracted_md' Markdown reduction for fast LLM consumption (optional cache)
--
-- All blobs live in the same content-addressed store as 'primary' does today —
-- only the (item_id, kind) → content_hash mapping is new.
--
-- items.content_hash stays as-is for backward compatibility; the 'primary'
-- row in item_artifacts is a redundant view of it that future code can treat
-- uniformly with other kinds. We backfill it below so existing queries don't
-- have to special-case "did the backfill happen".

CREATE TABLE item_artifacts (
    item_id       INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK(kind IN (
                      'primary',
                      'grobid_tei',
                      'ocr_text',
                      'extracted_md'
                  )),
    content_hash  TEXT NOT NULL,
    source        TEXT,
    generated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    size_bytes    INTEGER,
    PRIMARY KEY(item_id, kind)
);

CREATE INDEX idx_artifacts_hash ON item_artifacts(content_hash);

-- Backfill: every item that currently carries a content_hash gets a matching
-- 'primary' artifact row. metadata_source carries to the artifact source so
-- we can tell zotero-imported PDFs apart from CLI-ingested ones later.
INSERT INTO item_artifacts(item_id, kind, content_hash, source, generated_at)
SELECT id, 'primary', content_hash, metadata_source, added_at
FROM items
WHERE content_hash IS NOT NULL;
