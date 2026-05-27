"""FastAPI application entrypoint for the Fraud Radar service."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_router
from app.config import get_settings
from app.fraud import initialize_explainer

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model + SHAP explainer once at startup.

    A missing artifact directory raises here, which surfaces as a clear
    "server failed to start" rather than a delayed 503 mid-request.
    """
    settings = get_settings()
    artifacts_dir = Path(settings.model_artifacts_dir)
    log.info("Initialising fraud explainer from %s", artifacts_dir)
    initialize_explainer(artifacts_dir)
    yield
    # No teardown hook needed — the booster + explainer live for the process.


app = FastAPI(
    title="Fraud Radar API",
    description="Real-time fraud detection for financial transactions.",
    version="0.1.0",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe endpoint."""
    return {"status": "ok"}
