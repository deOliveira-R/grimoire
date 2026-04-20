# Next-session plan — post-v1 ranked backlog

**As of 2026-04-20, HEAD = `e2b7342` (Add item_artifacts table + GROBID
full-text artifact pipeline).** 265 tests passing, mypy strict clean.
Phases 0–7 of the main plan are done; three manual oracles block v1.0 tag
([#1](https://github.com/deOliveira-R/grimoire/issues/1),
[#2](https://github.com/deOliveira-R/grimoire/issues/2),
[#3](https://github.com/deOliveira-R/grimoire/issues/3)).

This doc is a handoff: a fresh Claude Code session should be able to pick
any sub-session below and execute without rebuilding context from scratch.
Read from top to bottom before writing any code.

## What just happened (last session's summary)

- Shipped: README, web UI refinement, Zotero migration, Phases 5–7, and
  a new `item_artifacts` table + GROBID full-text artifact pipeline
  (commit `e2b7342`).
- Filed 18 open issues on `deOliveira-R/grimoire`. Ranked by effort ×
  impact with S / A / B / C / D tiers.
- Opened: [#16](https://github.com/deOliveira-R/grimoire/issues/16)
  citation graph, [#17](https://github.com/deOliveira-R/grimoire/issues/17)
  section-aware chunking (needs schema approval),
  [#18](https://github.com/deOliveira-R/grimoire/issues/18) OCR via
  artifact kind.
- Rodrigo flagged that he plans to retire Zotero in favor of grimoire,
  so UI polish matters more than it otherwise would for a v1.

## Decisions to confirm before coding (≤ 5 min)

1. **Schema for [#17](https://github.com/deOliveira-R/grimoire/issues/17)
   section-aware chunking.** Needs `ALTER TABLE chunks ADD COLUMN
   section TEXT`. Only touches `chunks` (not `items`, `item_relations`,
   or the embedding tables), so strictly *outside* the plan §10 rule-5
   fence — but confirm anyway before writing migration 004.

2. **GROBID docker availability.** [#16](https://github.com/deOliveira-R/grimoire/issues/16)
   + [#17](https://github.com/deOliveira-R/grimoire/issues/17) both
   depend on TEI artifacts existing for each item. Before Session 2,
   confirm `docker compose up grobid` has been run once and
   `grimoire artifacts build --kind grobid_tei` has populated artifacts
   for the current library (~15k papers, ~12–42 h wall time). If the
   backfill hasn't happened, Session 2 starts with it.

3. **Confirm the ranked order still feels right.** Rodrigo ranked S-tier
   quick wins first ([#4](https://github.com/deOliveira-R/grimoire/issues/4),
   [#11](https://github.com/deOliveira-R/grimoire/issues/11),
   [#8](https://github.com/deOliveira-R/grimoire/issues/8),
   [#5](https://github.com/deOliveira-R/grimoire/issues/5)), A-tier
   strategic next. If daily use has changed his priorities, re-order at
   the top.

## Parallel user-side work (no Claude involvement)

These unblock v1.0 tag. Rodrigo does them; Claude writes up results
into [`project_oracle_results.md`](../../../../.claude/projects/-Users-rodrigo-git-grimoire/memory/project_oracle_results.md)
memory after.

- **[#1](https://github.com/deOliveira-R/grimoire/issues/1)** — KOReader
  iPad OPDS roundtrip. 30 min.
- **[#2](https://github.com/deOliveira-R/grimoire/issues/2)** — MCP
  research-question walkthrough via Claude Code. ~30 min once three real
  OPAL questions are drafted.
- **[#3](https://github.com/deOliveira-R/grimoire/issues/3)** — 20-query
  recall@10 oracle. Needs Rodrigo to draft the queries file
  (`tools/phase2_queries.jsonl` format: `{query, gold_item_ids,
  notes}`); then Claude wires it into `tools/phase2_search_oracle.py`
  and runs against the full corpus with real SPECTER2 + BGE-M3 loaded.

---

## Session 1 — S-tier quick wins (~1 working day)

Ship all four in one session. Zero schema changes, zero new deps. Run
the full suite + mypy at each step; no "fix-later" residue.

### 1.1 — Invariant 7 + 10 tests ([#4](https://github.com/deOliveira-R/grimoire/issues/4))

**Files:** new [`tests/test_invariants.py`](../../../tests/test_invariants.py).

**What:**

- **Invariant 7** (`paper + DOI ⇒ venue IS NOT NULL`): aspirational. Some
  DOIs legitimately have no venue in Crossref. Write as a *flag* test —
  compute the violation rate, fail only above a threshold (start with
  5 %, tighten later):
  ```python
  def test_invariant_7_venue_backfill_rate(tmp_db):
      # seed mixed rows, then query
      violations = tmp_db.execute("""
          SELECT COUNT(*) FROM items
          WHERE item_type='paper' AND doi IS NOT NULL AND venue IS NULL
      """).fetchone()[0]
      total = tmp_db.execute("""
          SELECT COUNT(*) FROM items
          WHERE item_type='paper' AND doi IS NOT NULL
      """).fetchone()[0]
      assert total == 0 or violations / total <= 0.05
  ```
- **Invariant 10** (no orphan chunks): FK cascade enforces it at insert
  time; this is belt-and-suspenders. Use `PRAGMA foreign_keys=OFF` to
  force an orphan, re-enable, assert the query returns 0:
  ```python
  def test_invariant_10_no_orphan_chunks(tmp_db):
      tmp_db.execute("PRAGMA foreign_keys=OFF")
      tmp_db.execute(
          "INSERT INTO chunks(item_id, chunk_index, text) VALUES (9999, 0, 'x')"
      )
      tmp_db.execute("PRAGMA foreign_keys=ON")
      orphans = tmp_db.execute("""
          SELECT COUNT(*) FROM chunks c
          LEFT JOIN items i ON i.id = c.item_id
          WHERE i.id IS NULL
      """).fetchone()[0]
      # Real libraries should stay at 0. Force-inserted orphan = 1 here so
      # the test documents the shape of the check; real-DB assertion is 0.
      assert orphans == 1
  ```
  Adjust: probably want a second test that runs against a clean fixture
  and asserts `== 0`, plus the forced-orphan test above as a sanity
  check of the query itself.

**Acceptance:**

- [ ] `pytest tests/test_invariants.py` passes.
- [ ] Full suite still passes (266+ tests).
- [ ] Mypy strict clean.

**Effort estimate:** ~1 h.

### 1.2 — Sort by first author ([#11](https://github.com/deOliveira-R/grimoire/issues/11))

**Files:**
- [`src/grimoire/web/queries.py`](../../../src/grimoire/web/queries.py)
  (`_SORT_ORDERS` dict).
- [`src/grimoire/web/ui.py`](../../../src/grimoire/web/ui.py)
  (`_SORT_LABELS` dict).
- [`tests/test_web_ui.py`](../../../tests/test_web_ui.py) (extend
  `test_home_sort_*`).

**What:** Add the `"author"` sort key. Correlated subquery in the
ORDER BY — SQLite handles this at 15 k scale without issue but verify
with an `EXPLAIN QUERY PLAN` once real data is loaded:

```python
_SORT_ORDERS = {
    "added":    "i.added_at DESC, i.id DESC",
    "year":     "i.publication_year DESC NULLS LAST, i.id DESC",
    "year_asc": "i.publication_year ASC NULLS LAST, i.id ASC",
    "title":    "LOWER(i.title) ASC, i.id ASC",
    "author":   (
        "(SELECT LOWER(a.family_name) FROM item_authors ia "
        "JOIN authors a ON a.id = ia.author_id "
        "WHERE ia.item_id = i.id AND ia.role = 'author' "
        "ORDER BY ia.position LIMIT 1) ASC NULLS LAST, i.id ASC"
    ),
}
```

`_SORT_LABELS`: add `"author": "First author (A–Z)"`.

**Gotcha:** items with no authors sort to the end (`NULLS LAST`).
Items-with-editors-but-no-authors (edited volumes) sort nowhere useful;
accept that, don't add editor fallback — edited volumes are a small
slice.

**Acceptance:**

- [ ] `GET /?sort=author` returns items ordered by first author family
      name.
- [ ] Sort dropdown renders "First author (A–Z)" as a selectable option.
- [ ] Test: seed items with first authors Z, A, M → order is A, M, Z
      after `?sort=author`.
- [ ] No regressions on other sort modes.

**Effort estimate:** ~1 h.

### 1.3 — Search term highlighting ([#8](https://github.com/deOliveira-R/grimoire/issues/8))

**Files:**
- New [`src/grimoire/web/jinja_filters.py`](../../../src/grimoire/web/jinja_filters.py).
- [`src/grimoire/web/ui.py`](../../../src/grimoire/web/ui.py) register
  filter on the `Jinja2Templates` env.
- [`src/grimoire/web/templates/home.html`](../../../src/grimoire/web/templates/home.html)
  wrap title + abstract.
- [`src/grimoire/web/templates/base.html`](../../../src/grimoire/web/templates/base.html)
  `<mark>` CSS.
- [`tests/test_web_ui.py`](../../../tests/test_web_ui.py) new test.

**What:**

```python
# jinja_filters.py
import re
from markupsafe import Markup, escape
from grimoire.search.keyword import _TOKEN

def highlight(text: str | None, query: str | None) -> Markup:
    """Wrap query terms in <mark>. Case-insensitive. HTML-escapes the
    input first, then injects <mark> around matches so the output is
    safe to render with |safe."""
    if not text or not query:
        return Markup(escape(text or ""))
    tokens = _TOKEN.findall(query)
    if not tokens:
        return Markup(escape(text))
    pattern = re.compile(
        "(" + "|".join(re.escape(t) for t in tokens) + ")",
        re.IGNORECASE,
    )
    escaped = str(escape(text))
    highlighted = pattern.sub(r"<mark>\1</mark>", escaped)
    return Markup(highlighted)
```

Register in `ui.py`:
```python
from grimoire.web.jinja_filters import highlight
templates.env.filters["highlight"] = highlight
```

Use in `home.html`:
```jinja
<h3><a href="/items/{{ item.item_id }}">{{ item.title|highlight(q) }}</a></h3>
...
<p class="abstract">{{ (item.abstract|truncate(280))|highlight(q) }}</p>
```

`<mark>` CSS in `base.html`:
```css
mark {
  background: rgba(255, 215, 0, 0.35);
  color: inherit;
  padding: 0 1px;
  border-radius: 2px;
}
@media (prefers-color-scheme: dark) {
  mark { background: rgba(255, 193, 7, 0.35); }
}
```

**Gotcha:** `truncate` before `highlight` means a match landing at the
truncation boundary can be elided. Acceptable — users see the full text
on the detail page. Truncate *after* highlight would chop inside
`<mark>` tags (character-based truncate isn't HTML-aware).

**Acceptance:**

- [ ] `GET /?q=boron` response contains `<mark>boron</mark>` in item
      titles/abstracts.
- [ ] Query terms in *any* case match: `q=Boron` also highlights
      `boron` and `BORON`.
- [ ] HTML-escape order is correct — no XSS with payload like
      `q=<script>`.
- [ ] No highlight rendered when `q` is empty (regression check on the
      default browse view).

**Effort estimate:** ~2 h.

### 1.4 — Nested collections tree ([#5](https://github.com/deOliveira-R/grimoire/issues/5))

**Files:**
- [`src/grimoire/web/queries.py`](../../../src/grimoire/web/queries.py)
  new `list_collections_tree()` returning `[TreeNode(collection, children=[...])]`.
- [`src/grimoire/web/ui.py`](../../../src/grimoire/web/ui.py) pass tree
  instead of flat list.
- [`src/grimoire/web/templates/home.html`](../../../src/grimoire/web/templates/home.html)
  replace the flat `<ul>` with a recursive macro.
- [`src/grimoire/web/templates/base.html`](../../../src/grimoire/web/templates/base.html)
  indent + toggle CSS.
- [`tests/test_web_ui.py`](../../../tests/test_web_ui.py) new test
  covering parent-child render.

**What:**

1. **Queries:**

```python
@dataclass
class CollectionTreeNode:
    collection: CollectionRow
    children: list["CollectionTreeNode"] = field(default_factory=list)
    descendants_count: int = 0  # rolled-up item count incl. children

def list_collections_tree(conn) -> list[CollectionTreeNode]:
    flat = list_collections(conn)
    by_id = {c.collection_id: CollectionTreeNode(c) for c in flat}
    roots: list[CollectionTreeNode] = []
    for c in flat:
        node = by_id[c.collection_id]
        if c.parent_id and c.parent_id in by_id:
            by_id[c.parent_id].children.append(node)
        else:
            roots.append(node)
    # Roll up counts bottom-up
    def _roll_up(node: CollectionTreeNode) -> int:
        node.descendants_count = node.collection.item_count + sum(
            _roll_up(ch) for ch in node.children
        )
        return node.descendants_count
    for r in roots:
        _roll_up(r)
    # Sort: alpha within each level
    def _sort(node):
        node.children.sort(key=lambda ch: ch.collection.name.lower())
        for ch in node.children:
            _sort(ch)
    roots.sort(key=lambda n: n.collection.name.lower())
    for r in roots:
        _sort(r)
    return roots
```

2. **Template** (recursive macro with `<details>` for toggle — no JS
needed if we accept one caveat):

```jinja
{% macro render_tree(node, depth, active_id) %}
  {%- set has_children = node.children|length > 0 -%}
  {%- set is_self_active = active_id == node.collection.collection_id -%}
  {%- set has_active_desc = has_descendant_active(node, active_id) -%}
  <li class="col-node" style="padding-left: {{ depth * 0.8 }}rem;">
    {% if has_children %}
      <details {% if has_active_desc or is_self_active %}open{% endif %}>
        <summary class="col-row">
          <a href="{{ qs_with({'collection': node.collection.collection_id}) }}"
             class="{% if is_self_active %}active{% endif %}"
             title="{{ node.collection.name }}">{{ node.collection.name }}</a>
          <span class="count">{{ node.descendants_count }}</span>
        </summary>
        <ul>
          {% for child in node.children %}
            {{ render_tree(child, depth + 1, active_id) }}
          {% endfor %}
        </ul>
      </details>
    {% else %}
      <div class="col-row">
        <a href="..." class="...">{{ node.collection.name }}</a>
        <span class="count">{{ node.collection.item_count }}</span>
      </div>
    {% endif %}
  </li>
{% endmacro %}
```

Caveat: `<details>`/`<summary>` interaction with clickable `<a>`
inside `<summary>` is weird on some browsers — clicking the link also
toggles. Mitigation: style the `▸/▾` marker as a separate affordance
(leave the default disclosure triangle) and instruct the user it toggles
the tree; the link still filters. Rodrigo's primary browsers (Safari,
Firefox) both handle this fine.

3. **Register Jinja helper** for `has_descendant_active`:

```python
def has_descendant_active(node, active_id):
    if active_id is None: return False
    if node.collection.collection_id == active_id: return True
    return any(has_descendant_active(ch, active_id) for ch in node.children)

templates.env.globals["has_descendant_active"] = has_descendant_active
```

4. **CSS** — add to `base.html`:

```css
aside.sidebar .col-node { list-style: none; }
aside.sidebar details > summary {
  display: flex; align-items: baseline; gap: .25rem;
  cursor: pointer; list-style: revert;
}
aside.sidebar details > summary::-webkit-details-marker {
  display: none;  /* we use the ▸/▾ from the default CSS list-style */
}
aside.sidebar details ul { padding-left: 0; margin: 0; }
```

**Acceptance:**

- [ ] Tree renders with proper indent depth.
- [ ] Each parent shows rolled-up count including descendants.
- [ ] Clicking a collection filters to it (URL picks up `?collection=N`).
- [ ] The tree auto-expands the path to the active collection.
- [ ] Test: seed nested collections (A > B > C with items in C), render,
      assert all three names appear and C-filter URL works.
- [ ] Zotero library with 40+ flat collections still renders fine (the
      tree-building code gracefully handles a flat structure).

**Effort estimate:** ~2–3 h.

### 1.5 — Session 1 closeout

After all four are green:

```bash
.venv/bin/python -m pytest --ignore=tests/test_heavy_embed.py -q    # expect 270+ passing
.venv/bin/python -m mypy src/grimoire                               # expect clean
```

One commit for the four: `"UI quick wins: sort-by-author + search highlighting + nested collections + invariant coverage"`. Push to
`origin/main`.

Close issues [#4](https://github.com/deOliveira-R/grimoire/issues/4),
[#5](https://github.com/deOliveira-R/grimoire/issues/5),
[#8](https://github.com/deOliveira-R/grimoire/issues/8),
[#11](https://github.com/deOliveira-R/grimoire/issues/11) via `gh issue close -c "landed in <sha>"`.

---

## Session 2 — A-tier strategic features (~1 working day)

### 2.1 — One-time GROBID backfill (preconditions for 2.2 and 4.x)

Before writing code:

```bash
docker compose up -d grobid    # ~1.5 GB image pull first run
curl http://localhost:8070/api/isalive   # expect 200

export GRIMOIRE_GROBID_URL=http://localhost:8070
grimoire artifacts status                 # see how many items still need TEI
grimoire artifacts build --kind grobid_tei -j 8    # parallel, resumable
```

**Expected wall time:** 3–10 s per paper × 15 k ≈ 12–42 h. Kick off and
come back tomorrow. Resumable via `items_missing_kind` — re-running
picks up where it left off.

Once complete: `grimoire artifacts status` should show `grobid_tei` with
~15 k rows.

### 2.2 — Citation graph ([#16](https://github.com/deOliveira-R/grimoire/issues/16))

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
    """Walk grobid_tei artifacts, DOI-match references → items,
    populate cites/cited_by edges."""
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

Ask Claude Code (via MCP) a trace-forward question to smoke-test end-to-end:
*"What papers in my library cite [some known paper]?"* — expect a
real answer backed by `list_related`.

---

## Session 3 — UI refinement pair (~1 working day)

Ship [#7](https://github.com/deOliveira-R/grimoire/issues/7) keyboard
nav and [#9](https://github.com/deOliveira-R/grimoire/issues/9)
split-pane HTMX together — they compound. Keyboard j/k without the
detail pane is meh; the detail pane without keyboard nav feels
mouse-only.

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

## Session 4 — Section-aware chunking ([#17](https://github.com/deOliveira-R/grimoire/issues/17))

**Preconditions:**
- Session 1 + 2 landed (S-tier quick wins + citation graph).
- GROBID backfill complete.
- **Schema approval obtained** (see "Decisions to confirm" above).

### 4.1 — Migration 004

New [`migrations/004_chunks_section.sql`](../../../migrations/004_chunks_section.sql):

```sql
ALTER TABLE chunks ADD COLUMN section TEXT;
```

Nothing else — old chunks get `NULL`, which MCP search treats as
"unknown section, don't filter out".

### 4.2 — Section classifier

New [`src/grimoire/section.py`](../../../src/grimoire/section.py):

```python
_HEADING_MAP: list[tuple[str, str]] = [
    ("introduction", "introduction"),
    ("background",   "introduction"),
    ("related work", "introduction"),
    ("method",       "methods"),
    ("methodolog",   "methods"),
    ("experimental", "methods"),
    ("materials",    "methods"),
    ("result",       "results"),
    ("finding",      "results"),
    ("discussion",   "discussion"),
    ("conclusion",   "conclusion"),
    ("summary",      "conclusion"),
]

def classify(heading: str | None) -> str:
    if not heading:
        return "other"
    h = heading.lower().strip()
    # Strip leading numbering like "1.", "2.3", "III."
    import re
    h = re.sub(r"^([IVX]+\.?\s+|\d+(\.\d+)*\.?\s+)", "", h)
    for needle, section in _HEADING_MAP:
        if needle in h:
            return section
    return "other"
```

Tests: headings with numbering ("2. Methods"), compound headings
("Materials and Methods"), foreign-language ("Méthodes" → "other", fine).

### 4.3 — Index flow rework

In [`index.py::index_item`](../../../src/grimoire/index.py):

Current flow builds `pages: list[tuple[int, str]]` and calls
`chunk_pages(pages)`. Rework:

```python
# Try TEI-backed section chunking first.
tei_bytes = artifacts.read(conn, item_id, "grobid_tei")
if tei_bytes is not None:
    struct = tei_parser.parse_structure(tei_bytes)
    if struct is not None and struct["sections"]:
        chunks = _chunk_tei_sections(struct["sections"])
        # Insert with section tag.
        ...
        return
# Fall back to per-page chunking (no section tag).
pages = _extract_pages(row["content_hash"])
chunks = chunk_pages(pages)
# Insert with section=None.
...
```

Implementation of `_chunk_tei_sections`:

```python
def _chunk_tei_sections(sections: list[dict]) -> list[tuple[Chunk, str]]:
    """Chunk within each section independently; tag each chunk with the
    classified section type."""
    out: list[tuple[Chunk, str]] = []
    for sec in sections:
        section_type = classify(sec["heading"])
        # Build a fake [(page=1, text)] input for the existing chunker —
        # page numbers are meaningless for TEI-sourced text, but our DB
        # accepts NULL on chunks.page so we use None directly via a small
        # variant of chunk_pages.
        single_page = [(None, sec["text"])]   # may need chunk.py tweak to accept None
        for c in chunk_pages(single_page):
            out.append((c, section_type))
    return out
```

Adjust `chunk.chunk_pages` to accept `page: int | None` — minor typing
change, no behavior change. Fine to land with this work.

Update the insert in `_insert_chunks_with_embeddings`:

```python
conn.execute(
    "INSERT INTO chunks(item_id, chunk_index, page, text, section) VALUES (?,?,?,?,?)",
    (item_id, chunk.chunk_index, chunk.page, chunk.text, section_type),
)
```

### 4.4 — MCP search `section` filter

In [`mcp/tools.py::search`](../../../src/grimoire/mcp/tools.py):

```python
def search(
    conn,
    query,
    mode="hybrid",
    item_types=None,
    section: str | None = None,    # new
    limit=20,
    *,
    item_embedder=None,
    chunk_embedder=None,
):
    ...
    # When `section` is set, filter chunk-search results to chunks tagged
    # with that section before hydrating. Item-level SPECTER2 results are
    # unaffected.
```

Wire through the FastMCP tool wrapper in `server.py`.

### 4.5 — Tests

- `test_section.py` — classifier covers each mapped keyword and the
  numbered-prefix strip.
- `test_index_section_tagging.py` — synthetic TEI + seeded item →
  `grimoire index` (with stub embedders) → assert `chunks.section`
  correctly populated per chunk.
- `test_mcp_search_section_filter.py` — store chunks with different
  sections, call search with `section='methods'`, assert only those
  chunks surface in snippets.

### 4.6 — Re-index

After landing, the user runs:
```bash
grimoire index --force
```
to re-chunk the full library with the new section tags. One overnight
run for the BGE-M3 portion.

### 4.7 — Session 4 closeout

Commit: `"Section-aware chunking from TEI artifacts (migration 004)"`.
Push. Close [#17](https://github.com/deOliveira-R/grimoire/issues/17).

Nice-to-have regression check: re-run
[`tools/phase2_search_oracle.py`](../../../tools/phase2_search_oracle.py)
with the new chunking, compare recall@10 on method-specific queries
against the pre-landing baseline. Update
[`project_oracle_results.md`](../../../../.claude/projects/-Users-rodrigo-git-grimoire/memory/project_oracle_results.md).

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
- **[#18](https://github.com/deOliveira-R/grimoire/issues/18)** OCR via
  artifact kind. Measure scan prevalence first — if <5 % of the library
  is scanned, deprioritize.
- **[#12](https://github.com/deOliveira-R/grimoire/issues/12)** Upload
  + URL-submit web forms. User explicitly deferred; revisit if
  non-CLI users need to ingest.
- **[#13](https://github.com/deOliveira-R/grimoire/issues/13)** v2
  umbrella — CSL styles, annotations, tag hierarchies, author
  disambiguation, bookmarklet. Revisit in v1.1+ when priorities clarify.

---

## Risks and watch-items across sessions

- **Prompt cache cost.** A plan this detailed fits in one session easily
  — don't bother breaking up for cache reasons.
- **Schema churn.** Only migration 004 (Session 4) touches `chunks`.
  Everything else is additive on existing schema.
- **GROBID backfill duration.** Sessions 2 and 4 depend on it. If it's
  not done when Session 2 starts, fall back to doing Session 1 now + a
  second sub-session of Session 2 later after the backfill completes.
- **Full-suite latency.** Currently ~3 s. Stays that way for Session 1.
  Session 4's migration 004 + re-chunking test fixtures will add maybe
  0.5 s. Still fine.
- **Mypy strict**: all new code must stay in. No `# type: ignore`
  without a comment explaining why.

---

## Suggested session structure

### Session 1 (1 day, pure Claude work)
- 10 min — read this doc top-to-bottom.
- 1 h   — #4 invariant tests.
- 1 h   — #11 sort by author.
- 2 h   — #8 search highlighting.
- 3 h   — #5 nested collections.
- 30 min — full-suite run, mypy, commit, push, close issues.

### Session 2 (same day the GROBID backfill finishes)
- 10 min — confirm backfill complete via `grimoire artifacts status`.
- 3 h   — #16 citation graph + tests.
- 30 min — smoke via MCP.
- 20 min — commit, push, close.

### Session 3 (1 day, pure Claude work)
- 30 min — template fragment split for partial rendering.
- 30 min — htmx + layout changes.
- 2 h   — keyboard nav.
- 1 h   — live browser smoke (user must be available for feedback).
- 30 min — commit, push, close.

### Session 4 (1 day, needs schema approval first)
- 10 min — confirm schema approval.
- 30 min — migration 004.
- 1 h   — section classifier + tests.
- 2 h   — index flow rework + tests.
- 1 h   — MCP search `section` filter + test.
- Overnight — `grimoire index --force` re-chunk.
- Next day, 30 min — Phase 2 oracle rerun for regression check.

**Total ≈ 4 focused working days** from "post-v1 tag" to "everything
important is landed."
