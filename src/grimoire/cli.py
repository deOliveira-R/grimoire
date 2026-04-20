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


if __name__ == "__main__":
    app()
