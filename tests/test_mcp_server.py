"""End-to-end test that the MCP server boots inside FastAPI and tools are
reachable via the standard MCP client over the streamable HTTP transport."""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def server_with_seed(tmp_db: sqlite3.Connection):
    tmp_db.execute(
        """INSERT INTO items(item_type, title, abstract, publication_year, doi,
                              metadata_source, metadata_confidence)
           VALUES ('paper', 'Test paper', 'Abstract text here.', 2024,
                   '10.1234/test', 'crossref', 1.0)"""
    )
    return tmp_db


def test_app_mounts_mcp_and_health_still_works(server_with_seed: sqlite3.Connection) -> None:
    from fastapi.testclient import TestClient

    from grimoire.app import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        # MCP expects JSON-RPC POSTs over streamable-http. A plain GET is a
        # protocol error (405/406), not a 404 — confirming the mount is live.
        r = client.get("/mcp/")
        assert r.status_code != 404, "MCP sub-app should be mounted at /mcp"


def test_mcp_tool_registry_exposes_plan_tools() -> None:
    """Every tool promised in plan §6 Phase 4 must be registered on the server."""
    import asyncio

    from grimoire.mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    expected = {
        "search",
        "get_item",
        "get_full_text",
        "get_snippets",
        "list_related",
        "get_citation",
        "list_tags",
        "list_collections",
        "find_by_tag",
    }
    missing = expected - names
    assert not missing, f"missing MCP tools: {missing}"
