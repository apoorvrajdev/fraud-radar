"""Application configuration via Pydantic Settings."""
from functools import lru_cache

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
    api_cors_origins: str = "http://localhost:5173"

    # ML
    model_path: str = "./ml/artifacts/model.joblib"
    preprocessor_path: str = "./ml/artifacts/preprocessor.joblib"

    # Simulator
    simulator_enabled: bool = True
    simulator_tx_per_second: int = 2


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance.

    Cached so the .env file is only parsed once per process.
    """
    return Settings()
