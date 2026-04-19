from __future__ import annotations

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


if __name__ == "__main__":
    app()
