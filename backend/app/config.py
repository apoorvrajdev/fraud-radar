"""Application configuration via Pydantic Settings."""
from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Environment variables override defaults. A .env file in the backend
    directory is auto-loaded for local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = "development"
    app_log_level: str = "INFO"

    # Database
    database_url: str = "sqlite:///./fraud_radar.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # CORS — explicit list of allowed origins for the dashboard. A
    # comma-separated env var (e.g. ``CORS_ORIGINS=http://a,http://b``)
    # is split into a list by the validator below. No wildcard origins,
    # ever — even in dev.
    cors_origins: list[str] = ["http://localhost:5173"]

    # ML
    # Directory holding model.json, feature_list.json, threshold.json, and
    # metrics.json — produced by `python -m ml.train`.
    model_artifacts_dir: str = "./ml/artifacts"

    # Simulator
    simulator_enabled: bool = True
    simulator_tx_per_second: int = 2

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: Any) -> Any:
        """Accept a CSV string from the env and split into ``list[str]``.

        Pydantic Settings hands env values through as strings; without
        this validator ``CORS_ORIGINS=http://a,http://b`` would fail to
        parse into ``list[str]``. Lists passed in Python (e.g. test
        overrides) are returned untouched.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance.

    Cached so the .env file is only parsed once per process.
    """
    return Settings()
