# Next-session plan — post-v1 ranked backlog

**As of 2026-04-21, HEAD = `c55df0b` (Section-aware chunking from TEI
artifacts, migration 004).** 301 tests passing, mypy strict clean.
Phases 0–7 of the main implementation plan are done. Sessions 1 (UI
quick wins) and 4 (section-aware chunking) of this backlog landed
today — issues #4, #5, #8, #11, #17 closed.

This doc is a handoff: a fresh Claude Code session should be able to
pick any sub-session below and execute without rebuilding context from
scratch. Read from top to bottom before writing any code.

## What shipped today (2026-04-20 → 2026-04-21)

- **Session 1** (`d56df40`): "UI quick wins" — closes #4, #5, #8, #11.
  - Invariant 7 (paper+DOI ⇒ venue violation-rate) + invariant 10 (no
    orphan chunks) test coverage.
  - `?sort=author` on the home page; correlated subquery on
    `item_authors`.
  - `<mark>`-based search term highlighting with HTML-safe escaping.
  - Nested collections tree in the sidebar (`<details>`/`<summary>`
    with descendants_count rollup).
- **Session 4** (`c55df0b`): "Section-aware chunking" — closes #17.
  - Migration 004 adds `chunks.section`.
  - `grimoire.section` classifier.
  - `index._tei_section_chunks` prefers TEI when `grobid_tei` artifact
    exists; falls back to per-page with `section=NULL`.
  - MCP `search` gains `section` parameter; item ranking unaffected.
- **OCR smoke test** on `1982NSE80-481.pdf` (56 pages, 1982 scan):
  `ocrmypdf` produces a clean searchable PDF in 38 s wall / 170 s CPU.
  Findings and a refined design (`ocr_pdf` artifact kind instead of
  `ocr_text`) posted as a comment on #18 (not priority-reclassified —
  that's Rodrigo's call).

## Decisions to confirm before coding (≤ 5 min)

1. **GROBID backfill status.** Sessions 2 and 4 (already shipped, but
   its section-tagging only fires when TEI exists) depend on
   `grobid_tei` artifacts existing for each item. Before picking up
   Session 2, confirm `docker compose up grobid` has been run and
   `grimoire artifacts build --kind grobid_tei` has populated artifacts
   for the current library (~15 k papers, ~12–42 h wall time).

   Check:
   ```bash
   grimoire artifacts status                # expect ~15 k rows under grobid_tei
   ```
   If the backfill hasn't happened, Session 2 starts with kicking it
   off and then waiting.

2. **OCR priority (#18).** Rodrigo confirmed historical scans matter
   strategically (IAEA reports, 1980s critical reviews on collision
   probabilities, transport approximations) for ORPHEUS work. Volume
   is medium-low, relevance density is high. Question for the next
   session: do Sessions 5 (OCR) before or after Sessions 2 / 3? The
   smoke-test comment on #18 has a design sketch and scope estimate.

3. **Ranked order.** Sessions below are ordered by ~ readiness to
   land, not strict priority. If Rodrigo's daily use has changed what
   matters, re-order at the top.

## Parallel user-side work (no Claude involvement)

These unblock the v1.0 tag. Rodrigo does them; Claude writes up
results into
[`project_oracle_results.md`](../../../../.claude/projects/-Users-rodrigo-git-grimoire/memory/project_oracle_results.md)
memory after.

- **[#1](https://github.com/deOliveira-R/grimoire/issues/1)** — KOReader
  iPad OPDS roundtrip. 30 min.
  - **Deferred indefinitely while Rodrigo is traveling without the
    iPad.** Pick up on return.
- **[#2](https://github.com/deOliveira-R/grimoire/issues/2)** — MCP
  research-question walkthrough via Claude Code. ~30 min once three real
  OPAL / ORPHEUS research questions are drafted.
- **[#3](https://github.com/deOliveira-R/grimoire/issues/3)** — 20-query
  recall@10 oracle. Needs Rodrigo to draft the queries file
  (`tools/phase2_queries.jsonl` format: `{query, gold_item_ids, notes}`);
  then Claude wires it into `tools/phase2_search_oracle.py` and runs
  against the full corpus with real SPECTER2 + BGE-M3 loaded.

---

## Session 2 — Citation graph ([#16](https://github.com/deOliveira-R/grimoire/issues/16))

**Preconditions:** GROBID backfill complete (`grimoire artifacts status`
shows non-trivial `grobid_tei` rows).

### 2.1 — GROBID backfill (if not done yet)

```bash
docker compose up -d grobid           # ~1.5 GB image pull first run
curl http://localhost:8070/api/isalive  # expect 200

export GRIMOIRE_GROBID_URL=http://localhost:8070
grimoire artifacts status             # see how many items still need TEI
grimoire artifacts build --kind grobid_tei -j 8
```

**Expected wall time:** 3–10 s per paper × 15 k ≈ 12–42 h. Kick off and
come back tomorrow. Resumable via `items_missing_kind` — re-running
picks up where it left off.

Also: this auto-triggers the benefit of Session 4's section-aware
chunking. After the backfill completes, run
`grimoire index --force` to re-chunk with section tags on TEI-bearing
items. Overnight for BGE-M3.

### 2.2 — Citation graph build

**Files:**
- New [`src/grimoire/citations.py`](../../../src/grimoire/citations.py)
  — DOI normalization + build logic.
- [`src/grimoire/cli.py`](../../../src/grimoire/cli.py) — add
  `citations_app = typer.Typer()` with `build` + `status` commands.
- New [`tests/test_citations.py`](../../../tests/test_citations.py).

**Shape of citations.py:**

```python
import re, logging, sqlite3
from dataclasses import dataclass
from grimoire import dedup
from grimoire.extract import tei as tei_parser
from grimoire.storage import artifacts

log = logging.getLogger(__name__)

_DOI_PREFIX_RE = re.compile(r"^(https?://)?(dx\.)?doi\.org/", re.IGNORECASE)

def normalize_doi(doi: str) -> str:
    return _DOI_PREFIX_RE.sub("", doi or "").strip().lower().rstrip(".")

@dataclass(slots=True)
class CitationReport:
    items_scanned: int = 0
    edges_inserted: int = 0
    self_citations_skipped: int = 0
    unresolved_dois: int = 0   # DOI not in library
    no_doi: int = 0            # reference had no DOI at all

def build(conn: sqlite3.Connection, *, limit: int | None = None,
          force: bool = False) -> CitationReport:
    # Build DOI → item_id map ONCE for the whole pass.
    doi_to_id = {
        normalize_doi(r["doi"]): int(r["id"])
        for r in conn.execute(
            "SELECT id, doi FROM items WHERE doi IS NOT NULL"
        )
    }
    # Walk every item that has a TEI artifact.
    ids = [
        int(r["item_id"])
        for r in conn.execute(
            "SELECT item_id FROM item_artifacts WHERE kind='grobid_tei' ORDER BY item_id"
        )
    ]
    if limit is not None:
        ids = ids[:limit]

    # Skip items that already have any cites edge unless --force.
    if not force:
        already_done = {
            int(r["subject_id"])
            for r in conn.execute(
                "SELECT DISTINCT subject_id FROM item_relations WHERE relation='cites'"
            )
        }
        ids = [i for i in ids if i not in already_done]

    report = CitationReport()
    for item_id in ids:
        data = artifacts.read(conn, item_id, "grobid_tei")
        if data is None:
            continue
        struct = tei_parser.parse_structure(data)
        if struct is None:
            continue
        report.items_scanned += 1
        for ref in struct["references"]:
            doi = ref.get("doi")
            if not doi:
                report.no_doi += 1
                continue
            target_id = doi_to_id.get(normalize_doi(doi))
            if target_id is None:
                report.unresolved_dois += 1
                continue
            if target_id == item_id:
                report.self_citations_skipped += 1
                continue
            # INSERT OR IGNORE on the unique relation triple — idempotent.
            dedup.apply_link(conn, item_id, target_id, "cites", 1.0)
            report.edges_inserted += 1
    return report
```

**CLI wiring** (in `cli.py`):

```python
citations_app = typer.Typer(help="Citation graph operations.")
app.add_typer(citations_app, name="citations")

@citations_app.command("build")
def citations_build(
    limit: int | None = typer.Option(None, "--limit"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    from grimoire.citations import build

    conn = connect()
    apply_migrations(conn)
    report = build(conn, limit=limit, force=force)
    typer.echo(f"scanned:             {report.items_scanned}")
    typer.echo(f"edges inserted:      {report.edges_inserted}")
    typer.echo(f"self-cites skipped:  {report.self_citations_skipped}")
    typer.echo(f"unresolved DOIs:     {report.unresolved_dois}")
    typer.echo(f"references w/o DOI:  {report.no_doi}")

@citations_app.command("status")
def citations_status() -> None:
    conn = connect()
    apply_migrations(conn)
    n = conn.execute(
        "SELECT COUNT(*) FROM item_relations WHERE relation='cites'"
    ).fetchone()[0]
    typer.echo(f"cites edges: {n}")
```

**Tests** (`test_citations.py`):

- `test_normalize_doi` — covers `https://doi.org/`, `http://dx.doi.org/`,
  trailing punctuation, mixed case.
- `test_build_inserts_cites_and_cited_by` — seed A + B with DOIs, store
  synthetic TEI for A whose references cite B's DOI, call build, assert
  both rows in `item_relations`.
- `test_build_skips_self_citations` — TEI of item A with its own DOI in
  refs → no insert.
- `test_build_is_idempotent` — double-run, same edges count.
- `test_build_reports_unresolved` — TEI ref DOI not in library → counted
  in `unresolved_dois`.

**MCP check:** `list_related(item_id, kind='citations')` returns the
new edges with no code change — verify with a short test.

**Acceptance:**

- [ ] `grimoire citations build` produces a sensible report against the
      full Zotero-imported library.
- [ ] Self-cites filtered.
- [ ] Re-running is a no-op (`INSERT OR IGNORE` on relation triple).
- [ ] `list_related(item_id, kind='citations')` returns real edges via
      MCP.
- [ ] All tests green.

**Effort estimate:** ~3–4 h (plus the GROBID backfill wall time which
runs async).

### 2.3 — Session 2 closeout

Commit: `"Citation graph from TEI references"`. Push. Close
[#16](https://github.com/deOliveira-R/grimoire/issues/16).

Ask Claude Code (via MCP) a trace-forward question to smoke-test
end-to-end: *"What papers in my library cite [some known paper]?"* —
expect a real answer backed by `list_related`.

---

## Session 3 — UI refinement pair (~1 working day)

Ship [#7](https://github.com/deOliveira-R/grimoire/issues/7) keyboard
nav and [#9](https://github.com/deOliveira-R/grimoire/issues/9)
split-pane HTMX together — they compound. Keyboard j/k without the
detail pane is meh; the detail pane without keyboard nav feels
mouse-only.

**Precondition:** Rodrigo available in a browser for the live-smoke
step (30 min feedback loop). Not GROBID-dependent.

### 3.1 — Template split for partial rendering

Break [`item.html`](../../../src/grimoire/web/templates/item.html) into:
- `item.html` — full page (unchanged URL, for deep links).
- `_item_body.html` — fragment with just the `<article class="detail">`.

Make `item.html` `{% include "_item_body.html" %}` so the two stay in
sync.

Extend [`ui.py::item_detail`](../../../src/grimoire/web/ui.py):

```python
@router.get("/items/{item_id}", response_class=HTMLResponse)
def item_detail(
    item_id: int,
    request: Request,
    partial: bool = Query(False),   # new
    conn: ...,
) -> HTMLResponse:
    ...
    template = "_item_body.html" if partial else "item.html"
    return templates.TemplateResponse(request=request, name=template, context={...})
```

### 3.2 — Add htmx via CDN

In `base.html` `<head>`:
```html
<script src="https://unpkg.com/htmx.org@1.9.12"
        integrity="sha384-..."
        crossorigin="anonymous"></script>
```

Pin the integrity hash — look up the current htmx version's SRI at ship
time.

### 3.3 — Split-pane layout + row hx-get

In `home.html`, wrap the existing `section.items` in a flex container
with a right-hand `<aside id="detail-pane">`. Only show the pane above
760 px.

Update each item row in `home.html`:
```jinja
<li class="item-row"
    hx-get="/items/{{ item.item_id }}?partial=1"
    hx-target="#detail-pane"
    hx-push-url="/items/{{ item.item_id }}"
    hx-trigger="click"
    data-item-id="{{ item.item_id }}"
    tabindex="0">
  ...
</li>
```

Keep the inner `<h3><a href="...">` link as a fallback (middle-click,
right-click "open in new tab", mobile).

### 3.4 — Keyboard nav

Extend the existing `/` handler script in `home.html`:

```javascript
(() => {
  const rows = () => [...document.querySelectorAll('li.item-row')];
  let selected = -1;

  function selectRow(i) {
    const list = rows();
    if (list.length === 0) return;
    selected = Math.max(0, Math.min(i, list.length - 1));
    list.forEach((el, idx) => el.classList.toggle('selected', idx === selected));
    list[selected].scrollIntoView({ block: 'nearest' });
  }

  function openSelected() {
    const list = rows();
    if (selected < 0 || selected >= list.length) return;
    htmx.trigger(list[selected], 'click');
  }

  function inEditable() {
    return ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName);
  }

  document.addEventListener('keydown', (e) => {
    if (inEditable() && e.key !== 'Escape') return;
    switch (e.key) {
      case '/':      e.preventDefault(); document.querySelector('input[name="q"]').focus(); break;
      case 'j':      e.preventDefault(); selectRow(selected + 1); break;
      case 'k':      e.preventDefault(); selectRow(selected - 1); break;
      case 'Enter':  e.preventDefault(); openSelected(); break;
      case 'G':      e.preventDefault(); selectRow(rows().length - 1); break;
      case 'Escape': document.activeElement.blur(); break;
      // 'g g' and '?' overlay can wait — ship j/k/enter/slash/esc first.
    }
  });
})();
```

CSS for `.item-row.selected` in `base.html`:
```css
li.item-row.selected {
  background: var(--accent-bg);
  outline: 2px solid var(--accent);
  outline-offset: -2px;
}
```

### 3.5 — Tests

Tests are tricky for JS behavior. Scope:

- Server-side: `GET /items/{id}?partial=1` returns a body without the
  `<header class="topbar">` (i.e. just the fragment). One assertion.
- Server-side: `?partial=1` on a missing item still returns 404.
- Manual live smoke: click around, j/k, enter, esc. Add a checklist in
  the commit message rather than a test.

### 3.6 — Session 3 closeout

Commit: `"Split-pane detail (HTMX) + keyboard navigation"`. Push.
Close [#7](https://github.com/deOliveira-R/grimoire/issues/7) and
[#9](https://github.com/deOliveira-R/grimoire/issues/9).

---

## Session 5 — OCR pipeline ([#18](https://github.com/deOliveira-R/grimoire/issues/18)) — ~½ day

**Context from today's smoke test** (see
[#18 comment](https://github.com/deOliveira-R/grimoire/issues/18#issuecomment-4291432046)):

- `ocrmypdf` 17.4.1 + Tesseract handled a 1982 NSE scan cleanly: 56
  pages, 38 s wall, 235 k extractable chars, cover metadata (title,
  authors, venue, year) all machine-readable.
- Greek letters + math glyphs mangled; prose and DOIs fine. Good enough
  for GROBID's reference extraction.
- Install gotcha: fresh `brew install ocrmypdf` with recently-upgraded
  x265 needs `brew reinstall libheif`.

**Preconditions:**
- Rodrigo confirmed the `ocr_pdf` vs `ocr_text` design question (the
  comment proposes storing the searchable PDF, not plain text).
- Rodrigo approves migration 005 adding `'ocr_pdf'` to the
  `item_artifacts.kind` CHECK constraint. It's outside the plan §10
  rule-5 fence (touches `item_artifacts` only, not `items` /
  `item_relations` / embedding tables).

### 5.1 — Migration 005

New [`migrations/005_ocr_artifact_kind.sql`](../../../migrations/005_ocr_artifact_kind.sql):

```sql
-- Add 'ocr_pdf' to the item_artifacts.kind CHECK constraint.
-- SQLite doesn't support modifying a CHECK in place; rebuild the table.

PRAGMA foreign_keys = OFF;

CREATE TABLE item_artifacts_new (
    item_id       INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK(kind IN (
                      'primary',
                      'grobid_tei',
                      'ocr_pdf',
                      'ocr_text',
                      'extracted_md'
                  )),
    content_hash  TEXT NOT NULL,
    source        TEXT,
    generated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    size_bytes    INTEGER,
    PRIMARY KEY(item_id, kind)
);

INSERT INTO item_artifacts_new SELECT * FROM item_artifacts;
DROP TABLE item_artifacts;
ALTER TABLE item_artifacts_new RENAME TO item_artifacts;

PRAGMA foreign_keys = ON;
```

Keep `ocr_text` in the allowed kinds (cheap) for anyone who later
wants a plain-text companion; nothing writes to it yet.

Also update the `Kind` Literal in
[`src/grimoire/storage/artifacts.py`](../../../src/grimoire/storage/artifacts.py):

```python
Kind = Literal["primary", "grobid_tei", "ocr_pdf", "ocr_text", "extracted_md"]
```

### 5.2 — `grimoire.ocr` subprocess wrapper

New [`src/grimoire/ocr.py`](../../../src/grimoire/ocr.py):

```python
"""Run ocrmypdf on a PDF and return the resulting searchable-PDF bytes.

Uses a subprocess so ocrmypdf's Python API pulling a full image stack is
not required at import time. Caller handles CAS writes and DB rows."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


class OCRUnavailable(RuntimeError):
    """Raised when ocrmypdf is not on PATH."""


def have_ocr() -> bool:
    return shutil.which("ocrmypdf") is not None


def run(
    pdf_bytes: bytes,
    *,
    force: bool = False,
    language: str = "eng",
    timeout_s: float = 1800,
) -> bytes:
    """OCR the PDF, return the resulting searchable-PDF bytes.

    When ``force`` is False (default), ocrmypdf skips pages that already
    have a text layer; pass True to re-OCR."""
    if not have_ocr():
        raise OCRUnavailable("ocrmypdf not found on PATH. Install with: brew install ocrmypdf")

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.pdf"
        out_path = Path(td) / "out.pdf"
        in_path.write_bytes(pdf_bytes)

        cmd = ["ocrmypdf", "--quiet", "--language", language]
        if force:
            cmd.append("--force-ocr")
        else:
            cmd.append("--skip-text")
        cmd.extend([str(in_path), str(out_path)])

        try:
            subprocess.run(cmd, check=True, timeout=timeout_s, capture_output=True)
        except subprocess.CalledProcessError as exc:
            log.warning("ocrmypdf failed (rc=%s): %s", exc.returncode, exc.stderr.decode())
            raise
        except subprocess.TimeoutExpired:
            log.warning("ocrmypdf timed out after %ss", timeout_s)
            raise
        return out_path.read_bytes()
```

### 5.3 — Hook into `artifacts build`

[`src/grimoire/cli.py`](../../../src/grimoire/cli.py) already has the
`artifacts build --kind grobid_tei` subcommand. Extend it to accept
`--kind ocr_pdf`:

- Use `artifacts.items_missing_kind(conn, 'ocr_pdf')` to get the work
  list.
- For each item, read the `primary` artifact, feed to `ocr.run`, store
  the result as `ocr_pdf`.
- **Gate: only OCR when the primary PDF has no text layer.** Use a
  helper:
  ```python
  def primary_has_text(conn, item_id) -> bool:
      p = artifacts.path_for(conn, item_id, 'primary')
      if p is None: return False
      import pymupdf
      with pymupdf.open(p) as doc:
          return any(page.get_text().strip() for page in doc)
  ```
  Skip items where it returns True.

Keep the `-j N` parallelism pattern from the grobid_tei subcommand.
Tesseract is CPU-bound; reasonable default `j=os.cpu_count()//2`.

### 5.4 — Prefer OCR'd PDF downstream

Two downstream sites should prefer the `ocr_pdf` artifact when present:

1. **`index._extract_pages`**: replace `CAS(settings.files_root).path_for_hash(content_hash)`
   with a lookup that prefers the `ocr_pdf` artifact when the item has
   one. This means chunks for OCR'd items have real text (currently
   zero chunks).
2. **GROBID build** (`artifacts build --kind grobid_tei`): when fetching
   the PDF to POST, prefer `ocr_pdf` over `primary` so Tesseract's
   layer feeds GROBID.

### 5.5 — Optional extra + pyproject

In [`pyproject.toml`](../../../pyproject.toml):

```toml
[project.optional-dependencies]
ocr = []    # ocrmypdf is a system binary, not a Python dep; the extra
            # exists for marker/future ocrmypdf-python-api use. Document
            # in README that the binary must be installed separately.
```

Add an install note in [README.md](../../../README.md) about
`brew install ocrmypdf` (macOS) / `apt install ocrmypdf` (Debian) and
the `brew reinstall libheif` workaround if x265 was recently upgraded.

### 5.6 — Tests

- `test_ocr.py`:
  - `test_have_ocr` — asserts `shutil.which` is patched correctly.
  - `test_run_raises_without_ocrmypdf` — monkeypatch `have_ocr()` False.
  - `test_run_with_mock_subprocess` — monkeypatch `subprocess.run` to
    drop a fake PDF in the output path; assert bytes round-trip.
- `test_artifacts_build_ocr.py`:
  - Seed an item with an image-only PDF as `primary`.
  - Stub `ocr.run` to return a known bytes blob.
  - Call the CLI; assert `ocr_pdf` artifact row exists.
  - Assert already-text-layer items are skipped.
- Optional live smoke (not in CI):
  ```bash
  grimoire artifacts build --kind ocr_pdf --limit 1
  ```
  on a known scanned item, then `grimoire index --force <item_id>` and
  verify chunks were produced.

### 5.7 — Session 5 closeout

Commit: `"OCR pipeline: ocr_pdf artifact kind + CLI wiring (migration 005)"`.
Push. Close [#18](https://github.com/deOliveira-R/grimoire/issues/18).

User-side follow-up after merge:
```bash
grimoire artifacts build --kind ocr_pdf -j 4    # bounded by CPU
grimoire artifacts build --kind grobid_tei      # picks up OCR'd PDFs
grimoire index --force                          # re-chunks with section tags
```

---

## Later sessions (B-tier, sketch only)

Not detailed here — deal with these as they bubble up during daily use:

- **[#6](https://github.com/deOliveira-R/grimoire/issues/6)** Author
  facet in sidebar (top-20 by count; use existing filter pattern).
- **[#10](https://github.com/deOliveira-R/grimoire/issues/10)**
  Multi-select + batch ops — significant UI work; revisit after #9 +
  #7 land and the overall feel is better.
- **[#14](https://github.com/deOliveira-R/grimoire/issues/14)** GROBID
  `processFulltext` fallback for TOC-less books. Low impact unless a
  measurable fraction of the library's books lack TOCs.
- **[#15](https://github.com/deOliveira-R/grimoire/issues/15)** Tier-4
  review queue — useful during/after a full 15 k migration; defer
  until then.
- **[#12](https://github.com/deOliveira-R/grimoire/issues/12)** Upload
  + URL-submit web forms. User explicitly deferred; revisit if
  non-CLI users need to ingest.
- **[#13](https://github.com/deOliveira-R/grimoire/issues/13)** v2
  umbrella — CSL styles, annotations, tag hierarchies, author
  disambiguation, bookmarklet. Revisit in v1.1+ when priorities
  clarify.

---

## Risks and watch-items across sessions

- **GROBID backfill duration.** Sessions 2 and 5 (if OCR'd PDFs go
  through GROBID) depend on it. If it's not done when Session 2 starts,
  fall back to kicking it off and running a different session in the
  interim.
- **Schema churn.** Migration 005 (Session 5) touches `item_artifacts`
  via table rebuild — a more invasive migration than the
  column-addition in 004. Be careful if the user has historical data.
  Test migration 005 against a copy of the real DB before running on
  the live one.
- **Full-suite latency.** Currently ~3.4 s for 301 tests. Sessions 2
  and 5 will add maybe 1 s. Still fine.
- **Mypy strict**: all new code must stay in. No `# type: ignore`
  without a comment explaining why.
- **Prompt cache cost.** This plan fits in one session easily — don't
  break it up for cache reasons.

---

## Suggested session structure

### Session 2 (same day the GROBID backfill finishes)
- 10 min — confirm backfill complete via `grimoire artifacts status`.
- 3 h   — #16 citation graph + tests.
- 30 min — smoke via MCP.
- 20 min — commit, push, close.

### Session 3 (1 day, needs Rodrigo available in a browser)
- 30 min — template fragment split for partial rendering.
- 30 min — htmx + layout changes.
- 2 h   — keyboard nav.
- 1 h   — live browser smoke (Rodrigo must be present for feedback).
- 30 min — commit, push, close.

### Session 5 (½ day, needs migration 005 approval first)
- 10 min — confirm approval.
- 30 min — migration 005 + artifacts Kind literal update.
- 1 h   — `grimoire.ocr` module + unit tests.
- 1 h   — `artifacts build --kind ocr_pdf` CLI + `primary_has_text`
  gate + test with stubbed subprocess.
- 30 min — downstream hooks: `_extract_pages` + GROBID build prefer
  `ocr_pdf`.
- 30 min — README install note, pyproject extra, commit, push, close.

**Total remaining ≈ 2.5 focused working days** from post-today state
to "everything important is landed."
