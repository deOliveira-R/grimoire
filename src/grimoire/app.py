from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from grimoire import __version__
from grimoire.config import settings
from grimoire.mcp.server import mcp
from grimoire.web.files import router as files_router
from grimoire.web.opds import router as opds_router
from grimoire.web.ui import router as ui_router


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # The streamable-http MCP sub-app uses a session manager whose
    # task group must be entered before any request arrives.
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="Grimoire", version=__version__, lifespan=_lifespan)

# Mount the MCP server at /mcp. Claude Code connects via
#   { "url": "http://<host>:8000/mcp" }
app.mount("/mcp", mcp.streamable_http_app())

app.include_router(opds_router)
app.include_router(files_router)
app.include_router(ui_router)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "data_root": str(settings.data_root),
    }
