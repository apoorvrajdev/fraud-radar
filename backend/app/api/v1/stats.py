"""Dashboard aggregate endpoints — Phase 3E.

Three read-only GETs that the dashboard polls. All operate on a rolling
24-hour window ending at request time. Window and bucket sizes are
hardcoded for Phase 3E (see `docs/adr/PHASE_3E_DESIGN.md` decision 1);
the query-string literals exist for forward compatibility but reject
anything other than the documented values via FastAPI's standard 422.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.stats import StatsBreakdown, StatsOverview, StatsTimeseries
from app.services import stats as stats_service

router = APIRouter(prefix="/stats", tags=["stats"])


def _now_utc() -> datetime:
    """Indirection so tests can freeze time via ``dependency_overrides``."""
    return datetime.now(UTC)


@router.get("/overview", response_model=StatsOverview, summary="24h KPI overview")
def overview(
    db: Session = Depends(get_db),
    now: datetime = Depends(_now_utc),
) -> StatsOverview:
    return stats_service.get_overview(db, now=now)


@router.get(
    "/timeseries",
    response_model=StatsTimeseries,
    summary="Per-hour fraud rate + volume for the trailing 24h",
)
def timeseries(
    window: Literal["24h"] = Query(
        default="24h",
        description="Only '24h' is accepted in Phase 3E.",
    ),
    bucket: Literal["1h"] = Query(
        default="1h",
        description="Only '1h' is accepted in Phase 3E.",
    ),
    db: Session = Depends(get_db),
    now: datetime = Depends(_now_utc),
) -> StatsTimeseries:
    # `window` and `bucket` are validated by Literal — values are
    # intentionally unused inside the service for Phase 3E (24h/1h is
    # hardcoded).
    del window, bucket
    return stats_service.get_timeseries(db, now=now)


@router.get(
    "/breakdown",
    response_model=StatsBreakdown,
    summary="Top-10 dimension breakdown for the trailing 24h",
)
def breakdown(
    dimension: Literal["country"] = Query(
        default="country",
        description="Only 'country' is accepted in Phase 3E.",
    ),
    db: Session = Depends(get_db),
    now: datetime = Depends(_now_utc),
) -> StatsBreakdown:
    del dimension
    return stats_service.get_breakdown(db, now=now)
