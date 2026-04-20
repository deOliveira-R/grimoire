-- Add part_of / contains_part to the relation vocabulary so multi-volume
-- works (TAOCP, Feynman Lectures, IAEA Safety Standards series, etc.) can
-- be linked to a synthetic "set" parent item. See plan §6 Phase 6.
--
-- SQLite doesn't allow modifying a table-level CHECK constraint in place,
-- so we rebuild item_relations and copy existing rows over. FK checks are
-- disabled for the swap per the SQLite docs' table-rebuild recipe.

PRAGMA foreign_keys = OFF;

CREATE TABLE item_relations_v2 (
    subject_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK(relation IN (
        'preprint_of', 'published_as',
        'erratum_for', 'corrected_by',
        'chapter_of', 'contains_chapter',
        'part_of', 'contains_part',
        'later_edition_of', 'earlier_edition_of',
        'translates', 'translated_from',
        'cites', 'cited_by',
        'related'
    )),
    object_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    confidence REAL DEFAULT 1.0,
    PRIMARY KEY(subject_id, relation, object_id)
);

INSERT INTO item_relations_v2 SELECT * FROM item_relations;
DROP TABLE item_relations;
ALTER TABLE item_relations_v2 RENAME TO item_relations;

PRAGMA foreign_keys = ON;
