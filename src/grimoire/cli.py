from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from grimoire import __version__
from grimoire.config import settings
from grimoire.db import apply_migrations, connect

app = typer.Typer(help="Grimoire literature server CLI.", no_args_is_help=True)
migrate_app = typer.Typer(help="One-shot migrations from other libraries.")
app.add_typer(migrate_app, name="migrate")
artifacts_app = typer.Typer(help="Build / inspect per-item derived artifacts.")
app.add_typer(artifacts_app, name="artifacts")


@app.command()
def version() -> None:
    """Print the grimoire version."""
    typer.echo(__version__)


@app.command("init-db")
def init_db() -> None:
    """Create the database and apply all pending migrations."""
    conn = connect()
    new = apply_migrations(conn)
    if new:
        typer.echo(f"Applied migrations: {', '.join(new)}")
    else:
        typer.echo("Database already up to date.")
    typer.echo(f"Database: {settings.db_path}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI server (health + MCP at /mcp + OPDS once Phase 5 lands)."""
    import uvicorn

    uvicorn.run("grimoire.app:app", host=host, port=port, reload=reload)


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", help="stdio | streamable-http"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8001, "--port"),
) -> None:
    """Run just the MCP server (useful for Claude Desktop / testing).

    For production the MCP endpoint is mounted into `grimoire serve` at /mcp."""
    from grimoire.mcp.server import mcp as mcp_server

    if transport == "stdio":
        mcp_server.run()
    elif transport == "streamable-http":
        mcp_server.settings.host = host
        mcp_server.settings.port = port
        mcp_server.run(transport="streamable-http")
    else:
        raise typer.BadParameter("transport must be stdio | streamable-http")


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, resolve_path=True),  # noqa: B008
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
) -> None:
    """Ingest a file or directory (.pdf, .epub) into the library."""
    from grimoire.ingest import ingest_path

    conn = connect()
    apply_migrations(conn)

    results = ingest_path(conn, path, recursive=recursive)
    counts: Counter[str] = Counter(r.outcome for r in results)

    for r in results:
        tag = r.outcome.upper()
        item = f"→ item_id={r.item_id}" if r.item_id is not None else ""
        typer.echo(f"[{tag}] {r.source_path} {item}")

    typer.echo("")
    typer.echo("Summary: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))


@app.command()
def index(
    force: bool = typer.Option(False, "--force", help="Re-embed items that already have vectors."),
    limit: int | None = typer.Option(None, "--limit", help="Cap the number of items processed."),
) -> None:
    """Compute item and chunk embeddings. Requires the 'ml' extra."""
    from grimoire.embed.bge_m3 import BGEM3Embedder
    from grimoire.embed.specter2 import Specter2Embedder
    from grimoire.index import index_all

    conn = connect()
    apply_migrations(conn)

    item_embedder = Specter2Embedder()
    chunk_embedder = BGEM3Embedder()
    results = index_all(
        conn,
        item_embedder=item_embedder,
        chunk_embedder=chunk_embedder,
        force=force,
        limit=limit,
    )
    counts: Counter[str] = Counter(r.status for r in results)
    for r in results:
        typer.echo(f"[{r.status.upper()}] item_id={r.item_id} chunks={r.chunks}")
    typer.echo("")
    typer.echo("Summary: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))


@app.command("dedup-scan")
def dedup_scan(
    limit: int | None = typer.Option(None, "--limit", help="Cap items scanned"),
    semantic: bool = typer.Option(
        True, "--semantic/--no-semantic", help="Run tier-4 (needs ml extras)"
    ),
) -> None:
    """Dry-run: apply the tiered dedup algorithm to existing items, print
    candidate merges/links without mutating the DB."""
    from grimoire import dedup
    from grimoire.models import Author, Metadata

    conn = connect()
    apply_migrations(conn)

    emb = None
    judge = None
    if semantic:
        from grimoire.dedup_llm import judge as llm_judge
        from grimoire.embed.specter2 import Specter2Embedder

        emb = Specter2Embedder()
        judge = llm_judge

    query = "SELECT id, title, abstract, doi, arxiv_id, isbn, content_hash FROM items ORDER BY id"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()

    counts: Counter[str] = Counter()
    for row in rows:
        authors = [
            Author(family_name=r["family_name"], given_name=r["given_name"])
            for r in conn.execute(
                """SELECT a.family_name, a.given_name FROM item_authors ia
                   JOIN authors a ON a.id = ia.author_id
                   WHERE ia.item_id=? ORDER BY ia.position""",
                (row["id"],),
            )
        ]
        cand = Metadata(
            title=row["title"],
            abstract=row["abstract"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            isbn=row["isbn"],
            authors=authors,
        )
        decision = dedup.decide(
            conn,
            cand,
            content_hash=row["content_hash"],
            item_embedder=emb,
            llm_judge=judge,
            exclude_item_id=int(row["id"]),
        )
        counts[decision.outcome] += 1
        if decision.outcome in {"merge", "link"}:
            typer.echo(
                f"[{decision.outcome.upper():6s}] item_id={row['id']:>4d} → "
                f"target={decision.target_id:>4d}  reason={decision.reason}  "
                f"conf={decision.confidence:.3f}"
            )
    typer.echo("")
    typer.echo("Summary: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))


@app.command()
def search(
    query: str = typer.Argument(...),
    mode: str = typer.Option("hybrid", "--mode", help="hybrid | keyword | semantic"),
    limit: int = typer.Option(10, "--limit", "-n"),
) -> None:
    """Search the library. Semantic/hybrid modes require the 'ml' extra."""
    from grimoire.search import search_items

    if mode not in {"hybrid", "keyword", "semantic"}:
        raise typer.BadParameter(f"mode must be hybrid|keyword|semantic, got {mode!r}")

    conn = connect()
    apply_migrations(conn)

    item_embedder = None
    chunk_embedder = None
    if mode in {"semantic", "hybrid"}:
        from grimoire.embed.bge_m3 import BGEM3Embedder
        from grimoire.embed.specter2 import Specter2Embedder

        item_embedder = Specter2Embedder()
        chunk_embedder = BGEM3Embedder()

    hits = search_items(
        conn,
        query,
        mode=mode,  # type: ignore[arg-type]
        limit=limit,
        item_embedder=item_embedder,
        chunk_embedder=chunk_embedder,
    )

    if not hits:
        typer.echo("No results.")
        return

    for i, h in enumerate(hits, start=1):
        typer.echo(f"[{i}] item_id={h.item_id}  {h.title}  ({h.year or '—'})")
        if h.snippet:
            page = f" p.{h.snippet.page}" if h.snippet.page else ""
            excerpt = h.snippet.text[:300].replace("\n", " ")
            typer.echo(f"     {page}  {excerpt}")


@migrate_app.command("zotero")
def migrate_zotero(
    library_path: Path = typer.Option(  # noqa: B008
        Path.home() / "Documents/Biblioteca/#Zotero/zotero.sqlite",
        "--library-path",
        "-l",
        help="Path to zotero.sqlite.",
    ),
    storage_dir: Path = typer.Option(  # noqa: B008
        Path.home() / "Documents/Biblioteca/#Zotero/storage",
        "--storage-dir",
        "-s",
        help="Zotero storage/ directory containing attached PDFs.",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Cap the number of items to import."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would happen without writing to the grimoire DB."
    ),
    semantic: bool = typer.Option(
        False, "--semantic", help="Run tier-4 semantic dedup during import (slower; needs ml)."
    ),
) -> None:
    """Import items from a local Zotero SQLite library."""
    from grimoire.migrate.zotero import migrate

    conn = connect()
    apply_migrations(conn)

    item_embedder = None
    llm_judge = None
    if semantic:
        from grimoire.dedup_llm import judge as llm_judge_fn
        from grimoire.embed.specter2 import Specter2Embedder

        item_embedder = Specter2Embedder()
        llm_judge = llm_judge_fn

    report = migrate(
        conn,
        library_path=library_path,
        storage_dir=storage_dir,
        limit=limit,
        dry_run=dry_run,
        item_embedder=item_embedder,
        llm_judge=llm_judge,
    )

    typer.echo("")
    typer.echo(f"Zotero migration report (source: {library_path}):")
    typer.echo(f"  candidates:            {report.total_candidates}")
    if dry_run:
        typer.echo("  (dry-run — no rows written)")
        return
    typer.echo(f"  inserted:              {report.inserted}")
    typer.echo(f"  merged (tier-1 match): {report.merged}")
    typer.echo(f"  skipped (already in):  {report.skipped_already_imported}")
    typer.echo(f"  skipped (no title):    {report.skipped_no_metadata}")
    typer.echo(f"  PDFs stored in CAS:    {report.pdf_attachments_stored}")
    if report.failures:
        typer.echo(f"  failures:              {len(report.failures)}")
        for f in report.failures[:5]:
            typer.echo(f"    - {f}")


@artifacts_app.command("build")
def artifacts_build(
    kind: str = typer.Option(
        "grobid_tei",
        "--kind",
        help="Artifact kind to build. Currently only 'grobid_tei' is supported.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Regenerate artifacts that already exist."
    ),
    limit: int | None = typer.Option(None, "--limit"),
    workers: int = typer.Option(
        4, "--workers", "-j", help="Parallel GROBID requests (server-side limit applies)."
    ),
) -> None:
    """Generate derived artifacts for all items that are missing them.

    Example:
      GRIMOIRE_GROBID_URL=http://localhost:8070 \\
        grimoire artifacts build --kind grobid_tei -j 8
    """
    if kind != "grobid_tei":
        raise typer.BadParameter(f"unsupported kind {kind!r}; only 'grobid_tei' for now")
    # Narrow the str from Typer to the literal storage type expects.
    artifact_kind: "artifacts.Kind" = "grobid_tei"

    from concurrent.futures import Future, ThreadPoolExecutor, as_completed
    from grimoire.extract import grobid
    from grimoire.storage import artifacts
    from grimoire.storage.cas import CAS

    if not settings.grobid_url:
        typer.echo("GRIMOIRE_GROBID_URL is unset — set it to your running GROBID.")
        raise typer.Exit(code=2)

    conn = connect()
    apply_migrations(conn)

    if force:
        ids = [
            int(r["item_id"])
            for r in conn.execute(
                "SELECT item_id FROM item_artifacts WHERE kind = 'primary' ORDER BY item_id"
            ).fetchall()
        ]
    else:
        ids = artifacts.items_missing_kind(
            conn, artifact_kind, primary_only=True, limit=limit
        )
    if limit is not None:
        ids = ids[:limit]

    typer.echo(f"{len(ids)} item(s) to process with GROBID fulltext (workers={workers}).")
    if not ids:
        return

    cas = CAS(settings.files_root)

    # Stage 1: resolve each item to its on-disk PDF path (main thread; the
    # sqlite connection is not safe to share with workers). A ``str`` entry
    # is a skip reason; a ``Path`` entry means "run GROBID on this path".
    from pathlib import Path as _Path

    tasks: list[tuple[int, _Path | str]] = []
    for item_id in ids:
        h = artifacts.get_hash(conn, item_id, "primary")
        if h is None:
            tasks.append((item_id, "no-primary-artifact"))
            continue
        path = cas.path_for_hash(h)
        if not path.exists():
            tasks.append((item_id, "cas-blob-missing"))
            continue
        tasks.append((item_id, path))

    # Stage 2: submit GROBID calls to worker threads. Workers only touch the
    # filesystem + HTTP; they never see the DB connection.
    def _run(path: _Path) -> bytes | None:
        return grobid.extract_fulltext(path)

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_item: dict[Future[bytes | None], int] = {}
        for item_id, path_or_reason in tasks:
            if isinstance(path_or_reason, str):
                # Skip case — log + count immediately, no GROBID call.
                typer.echo(f"  [{item_id:>6}] {path_or_reason}")
                failed += 1
                continue
            future_to_item[pool.submit(_run, path_or_reason)] = item_id

        # Stage 3: drain futures, write DB rows back on the main thread.
        for fut in as_completed(future_to_item):
            item_id = future_to_item[fut]
            try:
                data = fut.result()
            except Exception as exc:
                typer.echo(f"  [{item_id:>6}] grobid-error: {exc}")
                failed += 1
                continue
            if data is None:
                typer.echo(f"  [{item_id:>6}] grobid-failed")
                failed += 1
                continue
            artifacts.store(
                conn, item_id, artifact_kind, data, source="grobid-fulltext"
            )
            typer.echo(f"  [{item_id:>6}] ok")
            ok += 1

    typer.echo("")
    typer.echo(f"Summary: {ok} ok, {failed} failed")


@artifacts_app.command("status")
def artifacts_status() -> None:
    """Count artifacts by kind."""
    conn = connect()
    apply_migrations(conn)
    rows = conn.execute(
        """SELECT kind, COUNT(*) AS n, SUM(size_bytes) AS bytes
           FROM item_artifacts GROUP BY kind ORDER BY kind"""
    ).fetchall()
    total_items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    typer.echo(f"items: {total_items}")
    for r in rows:
        size = r["bytes"] or 0
        size_mb = size / (1 << 20)
        typer.echo(f"  {r['kind']:<14} {r['n']:>6}  {size_mb:>8.1f} MB")


if __name__ == "__main__":
    app()
