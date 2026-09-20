"""Model and dataset identity — Phase 5G.

One read-only GET describing what the API is serving and which
evaluation tracks exist beside it. Read from the artifact files rather
than from the loaded booster, so it works before the explainer is
initialised and cannot 500 a dashboard whose other tiles are fine.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.model_info import ModelInfo
from app.services.model_info import get_model_info

router = APIRouter(prefix="/model", tags=["model"])


@router.get(
    "",
    response_model=ModelInfo,
    summary="What model and dataset the API is serving, and what it is not",
)
def model_info() -> ModelInfo:
    """Return the serving model's identity plus the benchmark tracks.

    Every benchmark entry carries `served`, so a client can state the
    production/benchmark boundary without having to know the project's
    history. Only the in-house synthetic track is served; nothing has
    been promoted.
    """
    return get_model_info()
