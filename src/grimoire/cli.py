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


if __name__ == "__main__":
    app()
