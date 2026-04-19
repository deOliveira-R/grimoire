from __future__ import annotations

from fastapi import FastAPI

from grimoire import __version__
from grimoire.config import settings

app = FastAPI(title="Grimoire", version=__version__)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "data_root": str(settings.data_root),
    }
