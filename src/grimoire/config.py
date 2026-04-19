from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRIMOIRE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_root: Path = Field(
        default=Path("./data"),
        description="Root dataset. TrueNAS prod: /mnt/tank/grimoire.",
    )
    translation_server_url: str = "http://translation-server:1969"

    crossref_mailto: str | None = Field(
        default=None,
        description="Email for Crossref polite pool. Without it, rate limit drops 10x.",
    )

    llm_judge_enabled: bool = False
    llm_judge_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str | None = None

    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        return self.data_root / "db" / "library.db"

    @property
    def files_root(self) -> Path:
        return self.data_root / "files"

    @property
    def models_root(self) -> Path:
        return self.data_root / "models"


settings = Settings()
