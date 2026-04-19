from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from grimoire import __version__
from grimoire.config import settings
from grimoire.db import apply_migrations, connect

app = typer.Typer(help="Grimoire literature server CLI.", no_args_is_help=True)


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
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("grimoire.app:app", host=host, port=port, reload=reload)


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


if __name__ == "__main__":
    app()
